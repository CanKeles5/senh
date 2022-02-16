import argparse
import json
import os
import soundfile as sf


def preprocess_one_dir(in_dir, out_dir, out_filename):
    """ Create .json file for one condition."""
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    wav_list.sort()
    for wav_file in wav_list:
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples = sf.SoundFile(wav_path)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + ".json"), "w") as f:
        json.dump(file_infos, f, indent=4)


def preprocess(in_dir, out_dir):
    """ Create .json files for all conditions."""
    speaker_list = ["mix_clean", "s1", "s2", "noise"]
    for data_type in ["tr", "cv", "tt"]:
        for spk in speaker_list:
            preprocess_one_dir(
                os.path.join(in_dir, data_type, spk),
                os.path.join(out_dir, data_type),
                spk,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WHAM data preprocessing")
    parser.add_argument(
        "--in_dir", type=str, default=None, help="Directory path of wham including tr, cv and tt"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Directory path to put output files"
    )
    args = parser.parse_args()
    print(args)
    preprocess(in_dir="wav16k/min", out_dir="wav16k/min")
