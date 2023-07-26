import json  
import numpy as np
import matplotlib.pyplot as plt


def extract_data(filename):
    results = {}
    step_times = []
    with open(filename, "r") as f:
        for line in f:
            if "step time" in line:
                step_time = line.strip().split(" ")[-1]
                step_times.append(round(float(step_time), 2))
            elif "optimizer size" in line:
                results["checkpoint_size"] = int(float(line.strip().split(" ")[-2]))
            elif "valid gaps" in line:
                gaps = line.strip().split("valid gaps:")[-1]
                gaps = eval(gaps)
                total_idle_time = 0
                for key, value in gaps.items():
                    total_idle_time += float(value) / 1000
                results["idle_time"] = round(total_idle_time, 3)

    results["step_times"] = step_times
    results["checkpoint_time"] = round(results["checkpoint_size"] * 8 * 8 / 80 / 1000, 3)
    results["gemini_idle_time"] = results["idle_time"] - results["checkpoint_time"]
    return results



def extract_all_models(models):
    results = {}
    for model in models:
        filename = f"../{model}/log_5B"
        results[model] = extract_data(filename)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return results



def plot_iteration_time(results, jump_lines=10, ds_lines=18):
    params = {'legend.fontsize': 22, 'font.size': 26}
    plt.rcParams.update(params)
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = results.keys()
    N = len(labels)

    ds, gemini = [], []
    ds_err, gemini_err = [], []

    for key, values in results.items():
        step_times = values["step_times"]
        ds_times = step_times[jump_lines: ds_lines]
        ds.append(np.mean(ds_times))
        ds_err.append(np.std(ds_times))

        gemini_times = step_times[ds_lines + 2: ]
        gemini.append(np.mean(gemini_times))
        gemini_err.append(np.std(gemini_times))

    width = 0.25
    ind = np.arange(N)

    p2 = ax.bar(ind - 0.54* width, ds, width, yerr=ds_err, label=r"No checkpoint", capsize=3)
    p1 = ax.bar(ind + 0.54*width , gemini, width, yerr=gemini_err, label=r"GEMINI", capsize=3)

    ax.set_ylabel('Iteration time (sec)')
    # ax1.set_xlabel('Models')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 20)
    # plt.grid(False)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(loc=2, bbox_to_anchor=(0.0, 1.0), frameon=True)
    plt.show()
    plt.savefig("iter.png")



def plot_idle_time(results):
    params = {'legend.fontsize': 22, 'font.size': 26}
    plt.rcParams.update(params)
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = results.keys()
    N = len(labels)

    ds_idle = []
    gemini_ckpt = []
    gemini_idle = []

    for key, values in results.items():
        ds_idle.append(values["idle_time"])
        gemini_ckpt.append(values["checkpoint_time"])
        gemini_idle.append(values["gemini_idle_time"])

    width = 0.25
    ind = np.arange(N)

    p1 = ax.bar(ind-0.55*width, ds_idle, width, label=r"Idle time no ckpt", color='silver')
    p4 = ax.bar(ind+0.5*width, gemini_ckpt, width, label=r"GEMINI cpkt time", color=u'#ff7f0e', hatch='/')
    p5 = ax.bar(ind+0.5*width, gemini_idle, width, bottom=gemini_ckpt, label=r"Idle time w. GEMINI", color='silver', hatch='\\')

    colors= [u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd']

    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel('Time (sec)')
    ax.set_ylim(0, 3.2)
    # ax.set_yticks([1, 3, 5, 7])

    plt.grid(False)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(loc=2, ncol=1, frameon=True, bbox_to_anchor=(0.00, 1.02))
    plt.show()
    plt.savefig("idle_time.png")


if __name__ == "__main__":
    models = ["GPT", "BERT", "RobertaLM"]
    results = extract_all_models(models)
    plot_iteration_time(results, jump_lines=8, ds_lines=18)
    plot_idle_time(results)