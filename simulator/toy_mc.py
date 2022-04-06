import numpy as np
import numba

MAX_PROPAGATION_TIME = 10**9

class NeutronVetoToyLightPropagator:

    def __init__(self,
                 size_tpc: float,
                 pmt_properties: np.ndarray,
                 source_position: np.ndarray,
                 ) -> None:
        """Class which computes the arrival time of photons to the
        specified list of PMTs from a given source position.

        :param size_tpc: Float representing the TPC radius
        :param pmt_properties: Array of the shape n x 4 containing the
            x/y/z information as well channel number for each PMT.
        :param source_position: n x 3 array containing the various
            source positions from which light should be emitted.
        """
        if source_position.ndim == 1:
            source_position = np.array([source_position])

        self.size_tpc = size_tpc
        self.pmt_properties = pmt_properties
        self.source_position = source_position
        self.light_propagation = np.zeros((len(source_position), len(pmt_properties)),
                                          dtype=self._light_travel_dtype())

        # Fill some dummy values, later search for minimal propagation
        # distance and time.
        self.light_propagation['distance'] = MAX_PROPAGATION_TIME
        self.light_propagation['tof'] = MAX_PROPAGATION_TIME

        self._mask_not_directly_illuminated = blocked_by_tpc(
            source_position,
            pmt_properties[:, :3],
            size_tpc,)

        self._get_directly_illuminated_pmt_properties(self.source_position,
                                                      self.pmt_properties,
                                                      self._mask_not_directly_illuminated,
                                                      self.light_propagation,
                                                      )
        self._get_scatter_illuminated_pmt_properties(self.pmt_properties,
                                                     self.light_propagation,
                                                     self._mask_not_directly_illuminated,
                                                     self.size_tpc,
                                                     )

    @staticmethod
    @numba.njit
    def _get_directly_illuminated_pmt_properties(source_positions,
                                                 pmt_properties,
                                                 _mask_not_directly_illuminated,
                                                 light_propagation,
                                                 ):
        """Computes distance and time of flight for photons which
        directly illuminate a PMT. Changes self.light_propagation
        inplace.
        """
        for ind, s in enumerate(source_positions):
            _distance_direct_light = distance_points(
                s,
                pmt_properties[~_mask_not_directly_illuminated[ind], :3]
            )
            _direct_time_of_flight = get_time_of_flight(
                _distance_direct_light,)

            lp = light_propagation[ind]
            lp['distance'][~_mask_not_directly_illuminated[ind]] =  _distance_direct_light
            lp['tof'][~_mask_not_directly_illuminated[ind]] =  _direct_time_of_flight

    @staticmethod
    def _get_scatter_illuminated_pmt_properties(pmt_properties,
                                                light_propagation,
                                                _mask_not_directly_illuminated,
                                                size_tpc,
                                                ):
        """Computes distance and time of flight for photons to PMTs
        which are not directly illuminated. Uses as approximation time
        of flight from directly illuminated PMTs to not directly
        illuminated ones. Changes self.light_propagation inplace.
        """
        for sp_ind in range(len(light_propagation)):
            light_propagation_sp = light_propagation[sp_ind]
            _mask_not_directly_illuminated_sp = _mask_not_directly_illuminated[sp_ind]
            directly_illuminated_pmts = pmt_properties[~_mask_not_directly_illuminated_sp]
            dir_ill_pmt_propagation_prop = light_propagation_sp[~_mask_not_directly_illuminated_sp]

            for pmt, prop in zip(directly_illuminated_pmts, dir_ill_pmt_propagation_prop):
                _is_blocked = blocked_by_tpc(np.array([pmt]),
                                             pmt_properties[:, :3],
                                             size_tpc,
                                             )
                _is_second_order_pmt = (~_is_blocked
                                        &  _mask_not_directly_illuminated_sp)
                _is_second_order_pmt = _is_second_order_pmt[0]

                _distance_light = distance_points(
                    pmt[:3],
                    pmt_properties[_is_second_order_pmt, :3]
                )
                _distance_light += prop['distance']

                _time_of_flight = get_time_of_flight(
                    _distance_light,)

                distances = light_propagation_sp['distance'][_is_second_order_pmt]
                distances = np.min((distances, _distance_light), axis=0)
                light_propagation_sp['distance'][_is_second_order_pmt] = distances

                tof = light_propagation_sp['tof'][_is_second_order_pmt]
                tof = np.min((tof, _time_of_flight), axis=0)
                light_propagation_sp['tof'][_is_second_order_pmt] = tof

    def generate_events(self, n_events, n_photons, spread, threshold):
        """Creates for each of the specified source positions the
        specified number of events.
        :param n_events: Number of events per source position.
        :param n_photons: Number of photons per event.
        :param spread: Spread of photon-charge distribution.
        :param threshold: Threshold to be used.
        :returns: 3 nump.ndarrys. The first array is an array of the
            shape n x m, where n represents the number of source
            positions and m the number of events to be simulated. It
            contains the number of photons simulated for each event at
            the given source position.
            The second, third and fourth array are single dimensional
            and have the length of the sum of the first offset array.
            The contain the photon arrival times, channel and charge.
        """
        offsets, times, channels = self._generate_events(
            len(self.source_position),
            n_events,
            self.light_propagation,
            n_photons)
        charge = self._get_charge(offsets, spread, threshold)
        return offsets, times, channels, charge

    @staticmethod
    @numba.njit
    def _generate_events(n_source_positions,
                         n_events,
                         light_properties,
                         n_photons=26):
        offsets = np.zeros((n_source_positions, n_events), dtype=np.int32)
        _buffer_length = n_source_positions*n_events*n_photons*2
        photon_times = np.zeros(_buffer_length, dtype=np.float32)
        photon_channels = np.zeros(_buffer_length, dtype=np.int16)


        offset = 0
        for ind_sp in range(n_source_positions):
            _offsets = offsets[ind_sp]
            _light_properties = light_properties[ind_sp]
            for e_i in range(n_events):
                times = get_photon_timing(n_photons,
                                          _light_properties['tof'].min(),
                                          decay_constant=60)
                channels = get_channel(times, _light_properties['tof'])
                _n_ph = len(times)
                times += np.random.normal(0, 1, _n_ph)

                photon_times[offset:offset+_n_ph] = times
                photon_channels[offset:offset+_n_ph] = channels
                _offsets[e_i] = _n_ph
                offset += _n_ph
        return offsets, photon_times[:offset], photon_channels[:offset]

    @staticmethod
    def _get_charge(offsets, spread, threshold):
        charge = np.random.normal(1, spread, np.sum(offsets))
        charge = np.clip(charge, threshold, None)
        return charge

    def make_sample(self,
                    offsets,
                    times,
                    channels,
                    areas,
                    time_bins=np.arange(0, 32.1, 4)):
        _offsets = offsets.flatten()
        res = np.zeros((len(_offsets),
                        len(self.pmt_properties),
                        len(time_bins)-1), dtype=np.float32)
        #     res[:] = 10**-7 #TODO is this really needed?
        return self._make_sample(_offsets, times, channels, areas, time_bins, res)

    @staticmethod
    @numba.njit
    def _make_sample(offsets, times, channels, areas, time_bins, res):
        _offset = 0
        for ind, _nph in enumerate(offsets):
            _times = times[_offset:_offset+_nph]
            _channel = channels[_offset:_offset+_nph]
            _areas = areas[_offset:_offset+_nph]
            res_i = res[ind]
            for ch, t, a in zip(_channel, _times, _areas):
                mask = time_bins[:-1] <= t
                mask &= time_bins[1:] > t
                if not np.any(mask):
                    continue
                res_i[ch][mask] +=a
            _offset += _nph
        return res

    def plot_light_propagation(self, index_source_position=0):
        """Function which illustrates neutron-veto and PMTs which are
        blocked by direct illumination and which are not.
        :param index_source_position: Index of the source position to be
            plotted.
        :return: matplotlib figure.
        """
        fig, axes = plt.subplots()
        axes.set_aspect('equal')
        axes.scatter(self.pmt_properties[:, 0],
                     self.pmt_properties[:, 1])
        plt_tpc_circle(radius=81.5)

        axes.set_xlabel('X [cm]')
        axes.set_ylabel('Y [cm]')
        axes.scatter(*self.source_position[index_source_position, :2],
                     marker='d')

        _not_directly_illuminated = self._mask_not_directly_illuminated[index_source_position]
        axes.scatter(self.pmt_properties[_not_directly_illuminated, 0],
                     self.pmt_properties[_not_directly_illuminated, 1],
                     color='red')
        _light_propagation = self.light_propagation[index_source_position]
        for i in range(len(self.pmt_properties)):
            plt.text(self.pmt_properties[i, 0],
                     self.pmt_properties[i, 1],
                     f'{_light_propagation["tof"][i]:0.1f} ns'
                     )
        plt.close()
        return fig

    def _light_travel_dtype(delf):
        dtype = [
            (('"Minimal" travel distance from souce to PMT', 'distance'), np.float64),
            (('"Minimal" time of flight from source to PMT', 'tof'), np.float64)
        ]
        return dtype


@numba.njit
def blocked_by_tpc(point_of_emission, recording_pmt, max_radius):
    """Checks whether a certain photon would collide with TPC cryostat.
    :param point_of_emission: Point where photons are emitted from.
    :param recording_pmt: PMT which is suppose to record photon.
    :param max_radius: Radius of TPC.
    :returns:
    """

    res = np.zeros((len(point_of_emission),
                    len(recording_pmt)), dtype=np.bool_)
    for sp_ind, sp in enumerate(point_of_emission):
        for rp_ind, rp in enumerate(recording_pmt):
            res[sp_ind][rp_ind] = _blocked_by_tpc(sp, rp, max_radius)
    return res


@numba.njit
def _blocked_by_tpc(point_of_emission, recording_pmt, max_radius):
    # Drop z dimension for now:
    _poe = np.zeros(3)
    _rp = np.zeros(3)
    _poe[:2] = point_of_emission[:2]
    _rp[:2] = recording_pmt[:2]

    _pointing_vector, distance = direction_of_sight(_poe,
                                                    _rp,
                                                    )

    # If emission direction is opposite to TPC cannot block.
    # Check if emission point is closer to center than after traveling
    # a bit:
    distance_source_center  = distance_points(_poe,
                                              np.array([0, 0, 0]),
                                              )
    distance_source_destination  = distance_points(
        _poe + _pointing_vector * 0.01,
        np.array([0, 0, 0]),
        )
    if distance_source_destination > distance_source_center:
        return False


    # now check if for other signal gets blocked:
    radius = distance_point_and_line(np.array([0, 0, 0]),
                                     _pointing_vector,
                                     _poe,
                                     )

    return radius < max_radius


@numba.njit
def direction_of_sight(pmt_a, pmt_b):
    """Equation for pointing vector of line of sight.
    """
    distance = distance_points(pmt_a, pmt_b)
    return (pmt_b - pmt_a)/distance, distance

@numba.njit
def distance_point_and_line(point, direction, starting_point):
    """Computes for a given line the distance to a point.
    :param point: Point to which distance should be computed.
    :param direction: Directional vector of the line.
    :param starting_point: Starting point of the line.
    :return: Distance between line and point.
    """
    vec_point_line = np.cross((point - starting_point), direction)
    return np.sqrt(np.sum(vec_point_line**2))


@numba.njit
def distance_points(A, B):
    """Computes distance between two points.
    """
    # Vectorize this function in A or B:
    # I did not manage with guvectorize...
    if B.ndim > 1:
        res = np.zeros(len(B))
        for i in range(len(B)):
            res[i] = _distance_points(A, B[i])
        return res

    if A.ndim > 1:
        res = np.zeros(len(A))
        for i in range(len(A)):
            res[i] = _distance_points(A[i], B)
        return res

    return _distance_points(A, B)


@numba.njit
def _distance_points(A, B):
    """Computes distance between two points.
    """
    return np.sqrt(np.sum((B - A)**2 ))


@numba.njit
def get_time_of_flight(distances, n=1.3):
    """Computes time of flight for photons onto directly illuminated
    PMTs.
    :param distances: Distance from source to PMT in cm.
    :param n: Refractive index of medium.
    :returns: Array with time of flight information in ns.
    """
    _distances = distances/100 # converting into m
    tof = _distances/(3*10**8/n)*10**9 # ns
    return tof


def get_pmt_source_positions(pmts, distance_to_pmt):
    """Function creating source positions/"position of light emission"
    with a certain distance to the PMTs.
    :param pmts: np.array storing the x and y information of the PMT
        positions.
    :distance_to_pmt: Radial distance from the PMT towards the detector
        center where the light should be emitted from.
    :returns: np.array of shape 2 x n with positions.
    """
    x = pmts[:, 0]
    y = pmts[:, 1]
    angle = np.angle(x+y*1j)

    r_source = np.sqrt(x**2 + y**2) - distance_to_pmt
    x_source = r_source * np.cos(angle)
    y_source = r_source * np.sin(angle)
    return np.array([x_source, y_source])


def get_random_cyrcle_position(radius, n_positions):
    """Function which draws n random positions on a circle.
    :param radius: Radius of the cyrcle
    :param n_positions: Number of positions to be drawn.
    :returns: np.array of the shape 2 x n with x/y positions
    """
    angle = np.random.uniform(-np.pi, np.pi, n_positions)
    x_source = radius * np.cos(angle)
    y_source = radius * np.sin(angle)
    return np.array([x_source, y_source])


def plt_tpc_circle(radius=81.5):
    """
    Plots circle with certain radius.
    """
    x = np.arange(-3*np.pi, 3*np.pi, 0.1)
    plt.plot(radius * np.sin(x),
             radius * np.cos(x), color='k')




@numba.njit
def get_photon_timing(av_n_ph, minimal_propagation_time, decay_constant=60):
    n_ph = np.random.poisson(lam=av_n_ph)
    times = np.random.exponential(60, n_ph) + minimal_propagation_time
    return times


@numba.njit
def get_channel(photon_times, light_propagation):
    res = np.zeros(len(photon_times), dtype=np.int16)
    for ind, t in enumerate(photon_times):
        ch = np.random.choice(pmts_in_z_slice[:, 3][light_propagation < t])
        res[ind] = ch
    return res
