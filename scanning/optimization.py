
class Simulation():

    pols_on = None     # which initial polarizations are on 
    # wafers_on, rhombi_on 

    px_list = None # performing stats on these pixels
    norm_pxan = None    

    def __init__(self, max_acc=None, min_vel=None, convolve=False, norm_time=False, pols_on=None, rhombi_on=None, wafers_on=None) -> None:
        # pass TelescopePattern, Instrument
        self._simulate_scan()
        self._stats()

    def _simulate_scan(self):
        sky_hist = None

        # for single-pixel analysis:
        det_hist = None
        time_hist = None

        # maybe more 

    def _stats(self):
        # generate metrics for optimization (if not already)
        pass

    @property
    def std_dev(self):
        pass

