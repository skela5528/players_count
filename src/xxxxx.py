from typing import List


class Runner:

    def get_detector(self):
        ...

    def get_player_counter(self):
        ...

    def save_results(self):
        ...

    def main(self):
        ...


if __name__ == '__main__':
    args = get_args()
    Runner.main()