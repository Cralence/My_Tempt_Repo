import os
from os.path import join as pjoin
import argparse
import time


# automatically create and continue jobs in the server

def system_call(cmd):
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--round",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    N = args.round
    out_dir = 'output'
    log_file_predix = 'trans_v10'
    cmd = 'sbatch run_transformer_v10_dcs.sh'

    for i in range(N):
        print(f'Training round {i}')

        out_dir_items = os.listdir(out_dir)
        out_dir_items = [s for s in out_dir_items if s[:len(log_file_predix)] == log_file_predix]
        for item in out_dir_items:
            system_call(f'rm {pjoin(out_dir, item)}')

        system_call(cmd)

        while True:
            time.sleep(1)
            out_dir_items = os.listdir(out_dir)
            out_dir_items = [s for s in out_dir_items if s[:len(log_file_predix)] == log_file_predix]
            if len(out_dir_items) != 0:
                # program started
                print('Program started!')
                break

        # if started, start to count time
        sleep_time = 6 * 3600 + 300
        time.sleep(sleep_time)





