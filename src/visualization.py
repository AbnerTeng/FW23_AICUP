import os
import plotly.express as px
from .preproc import PreProc

class Visualization(PreProc):
    def __init__(self) -> None:
        super().__init__(f'{os.getcwd()}/data/training_data.csv')
        self.data = super().main()


    def viz(self) -> None:
        """
        simple visualization
        """
        fig = px.scatter(self.data, x = '橫坐標', y = '縱坐標', color = '單價')
        fig.show()


    def main(self) -> None:
        self.viz()


if __name__ == "__main__":
    viz = Visualization()
    viz.main()
        