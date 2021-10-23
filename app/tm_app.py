import os

from flask import Flask, render_template
import glob
import shutil

app = Flask(__name__, static_folder='../datasets')


def traverse_dataset(dataset_route):
    data = []
    pairs = glob.glob(f'{dataset_route}/*')
    for pair in pairs:
        pair = pair.replace("\\", "/")
        idx = pair.split("/")[-1]
        A = glob.glob(f'{pair}/A/*')[0].replace("\\", "/")
        B = glob.glob(f'{pair}/B/*')[0].replace("\\", "/")
        ours_unprocessed = glob.glob(f'{pair}/ours/*')
        ours = []
        for our in ours_unprocessed:
            if "ours.png" in our:
                continue
            ours.append(our.replace("\\", "/"))

        data.append({
            'idx': idx,
            'A_src': A,
            'B_src': B,
            'ours': ours
        })

    return data


def set_ours(dataset_name, idx, result):
    result_path = f"../datasets/afhq/{dataset_name}/{idx}/ours/{result}.png"
    target_path = f"../datasets/afhq/{dataset_name}/{idx}/ours/ours.png"
    if os.path.exists(target_path):
        os.remove(target_path)
    shutil.copyfile(result_path, target_path)


@app.route("/cat2cat")
def dataset_page():
    dataset_name = "cat2cat"
    data_pairs = traverse_dataset(dataset_route="../datasets/afhq/cat2cat")
    return render_template('editor.html', data_pairs=data_pairs, dataset_name=dataset_name)


@app.route("/update_ours/cat2cat/<idx>/<result>")
def update_ours_cat2cat(idx, result):
    set_ours('cat2cat', idx, result)
    return "ok"
