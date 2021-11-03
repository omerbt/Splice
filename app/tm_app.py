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
        A_src = glob.glob(f'{pair}/A/*')[0].replace("\\", "/")
        B_src = glob.glob(f'{pair}/B/*')[0].replace("\\", "/")
        ours_src = f'{pair}/ours/ours.png'
        ours_unprocessed = glob.glob(f'{pair}/ours/*')
        ours = []
        for our in ours_unprocessed:
            if "ours.png" in our:
                continue
            ours.append(our.replace("\\", "/"))

        data.append({
            'idx': idx,
            'A_src': A_src,
            'B_src': B_src,
            'ours': ours,
            'ours_src': ours_src
        })

    return data


def set_ours_afhq(dataset_name, idx, result):
    result_path = f"../datasets/afhq/{dataset_name}/{idx}/ours/{result}.png"
    target_path = f"../datasets/afhq/{dataset_name}/{idx}/ours/ours.png"
    if os.path.exists(target_path):
        os.remove(target_path)
    shutil.copyfile(result_path, target_path)


def set_ours_landscape(idx, result):
    result_path = f"../datasets/landscape/{idx}/ours/{result}.png"
    target_path = f"../datasets/landscape/{idx}/ours/ours.png"
    if os.path.exists(target_path):
        os.remove(target_path)
    shutil.copyfile(result_path, target_path)


@app.route("/cat2cat")
def cat2cat_dataset_page():
    dataset_name = "cat2cat"
    data_pairs = traverse_dataset(dataset_route="../datasets/afhq/cat2cat")
    return render_template('editor.html', data_pairs=data_pairs, dataset_name=dataset_name)


@app.route("/dog2dog")
def dog2dog_dataset_page():
    dataset_name = "dog2dog"
    data_pairs = traverse_dataset(dataset_route="../datasets/afhq/dog2dog")
    return render_template('editor.html', data_pairs=data_pairs, dataset_name=dataset_name)


@app.route("/landscape")
def landscape_dataset_page():
    dataset_name = "landscape"
    data_pairs = traverse_dataset(dataset_route="../datasets/landscape")
    return render_template('editor.html', data_pairs=data_pairs, dataset_name=dataset_name)



@app.route("/cat2cat/view")
def cat2cat_dataset_view_page():
    data_pairs = traverse_dataset(dataset_route="../datasets/afhq/cat2cat")
    return render_template('viewer.html', data_pairs=data_pairs)


@app.route("/dog2dog/view")
def dog2dog_dataset_view_page():
    data_pairs = traverse_dataset(dataset_route="../datasets/afhq/dog2dog")
    return render_template('viewer.html', data_pairs=data_pairs)


@app.route("/landscape/view")
def landscape_dataset_view_page():
    data_pairs = traverse_dataset(dataset_route="../datasets/landscape")
    return render_template('viewer.html', data_pairs=data_pairs)


@app.route("/update_ours/cat2cat/<idx>/<result>")
def update_ours_cat2cat(idx, result):
    set_ours_afhq('cat2cat', idx, result)
    return "ok"


@app.route("/update_ours/dog2dog/<idx>/<result>")
def update_ours_dog2dog(idx, result):
    set_ours_afhq('dog2dog', idx, result)
    return "ok"


@app.route("/update_ours/landscape/<idx>/<result>")
def update_ours_landscape(idx, result):
    set_ours_landscape(idx, result)
    return "ok"
