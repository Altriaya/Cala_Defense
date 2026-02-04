import numpy as np

class TimeBasedSplitter:
    def __init__(self, steps_per_day=288):
        """
        Splits traffic data based on Time-of-Day semantics.
        PEMS 5-min data -> 288 steps/day.
        
        Environments:
        0: AM Peak (07:00 - 09:00) -> Indices [84, 108]
        1: PM Peak (17:00 - 19:00) -> Indices [204, 228]
        2: Off-Peak (Rest)
        """
        self.steps_per_day = steps_per_day
        self.am_range = (int(7 * 12), int(9 * 12))   # 84 - 108
        self.pm_range = (int(17 * 12), int(19 * 12)) # 204 - 228

    def split(self, data):
        """
        Args:
            data: (Total_Steps, N, C) - Raw sequential data preferred.
        """
        if data.ndim == 3:
            total_steps = data.shape[0]
            indices = np.arange(total_steps)
            tod = indices % self.steps_per_day
            
            mask_am = (tod >= self.am_range[0]) & (tod < self.am_range[1])
            mask_pm = (tod >= self.pm_range[0]) & (tod < self.pm_range[1])
            mask_off = ~(mask_am | mask_pm)
            
            return [data[mask_am], data[mask_pm], data[mask_off]]
        else:
            # Fallback for windowed data if index unavailable
            print("[Warning] Splitter received windowed/shuffled data. Returning random split fallback.")
            n = len(data)
            return [data[:n//3], data[n//3:2*n//3], data[2*n//3:]]

    def get_env_id(self, global_step):
        tod = global_step % self.steps_per_day
        if self.am_range[0] <= tod < self.am_range[1]:
            return 0 # AM
        elif self.pm_range[0] <= tod < self.pm_range[1]:
            return 1 # PM
        else:
            return 2 # Off

if __name__ == "__main__":
    pass
