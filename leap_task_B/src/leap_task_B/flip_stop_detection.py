import numpy as np

class FlipStopDetector:
    def __init__(self, threshold=0.01, verbose=False):
        self.prev_value = None
        self.increase_count = 0
        self.decrease_count = 0
        self.phase = None
        self.threshold = threshold
        self.verbose=verbose

    def clear_prev_value(self):
        self.increase_count = 0
        self.decrease_count = 0
        self.phase = None
        self.prev_value = None
    
    def process_data_point(self, value):
        if self.prev_value is not None:
            
            if value > self.prev_value + self.threshold:
                self.increase_count += 1
                self.decrease_count = 0
            elif value < self.prev_value - self.threshold:
                self.decrease_count += 1
                self.increase_count = 0
            # else:
            #     不归零
            #     self.increase_count = 0
            #     self.decrease_count = 0

            if self.increase_count >= 5:
                if self.phase == "down_phase":
                    self.phase = "up_phase"
                    if self.verbose: print("start to go up after going down")

                    return True
            elif self.decrease_count >= 5:
                if self.phase != "down_phase":
                    self.phase = "down_phase"
                    if self.verbose: print("start to go down")
                    return False
        
        self.prev_value = value
        return False