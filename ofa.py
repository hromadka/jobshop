# purpose of this module is to generate all JSSP schedules 
# using the OFA algorithm based on Cliff Stein
# 8/12/2024


import collections

def main():
    jobs_data = [  # task = (machine_id, processing_time).
        [(0, 10), (1, 8), (2, 4)],  # Job0
        [(1, 8), (0,3), (3, 5), (2, 6)],  # Job1
        [(0, 4), (1,7), (3, 3)],  # Job2
    ]




if __name__ == "__main__":
    main()