from .n_facilities import NFacilities

class MeanDist(NFacilities):
    """
    The class that can get the mean value of k nearest neighbors
    """
    def __init__(self, facility_path: str, target_path: str):
        #This class does not need the radius, so I set it to 0
        #Override functions that need the radius to prevent errors
        super().__init__(facility_path, target_path, 0)

    def mean(self, k=3):
        """
        Get the mean value of k nearest neighbors
        """
        facility_pos, target_pos = self.split_data()

        #sort by distance
        facility_pos = self.calculate_dist(facility_pos, target_pos)
        facility_pos = facility_pos.sort_values(by = 'distance', ascending=True)

        #return mean value of k nearest neighbors
        mean = facility_pos.iloc[0:k, 2].mean()
        return mean

    #Override the function that can not be used in this class
    def find_n_facilities(self):
        raise NotImplementedError("find_n_facilities can not use in MeanDist")

    def main(self):
        raise NotImplementedError("find_n_facilities can not use in MeanDist")

if __name__ == "__main__":
    TARGET_DATA = "data/small_training_data.csv"
    FACILITY_DATA = "data/external_data/金融機構基本資料.csv"

    md = MeanDist(FACILITY_DATA, TARGET_DATA)
    md.main()
    print(md.mean(k=5))
