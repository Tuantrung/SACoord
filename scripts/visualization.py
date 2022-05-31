import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# YOU NEED TO CHANGE THIS: copy the result path from above - without the trailing "result.yaml"
test_result_dir = "../results/SA_weighted-f05d05/5node-in1-rand-cap0-2/abc/det-arrival10_det-size001_duration100/" \
                  "2022-05-27_11-08-41_seed1234"

# read test results into pandas data frame
test_results = os.path.join(test_result_dir, "metrics.csv")
df = pd.read_csv(test_results)

# let's add a column with the percentage of successful flows (in [0,100]%)
df["success_perc"] = 100 * df["successful_flows"] / df["total_flows"]

X = df["time"]
Y1 = df["success_perc"]
Y2 = df["avg_end2end_delay"]

# create figure instance
fig1 = plt.figure(1)
fig1.set_figheight(11)
fig1.set_figwidth(8.5)

ax = fig1.add_subplot(2, 1, 1)
ax.plot(X, Y1)
plt.xlabel('time')
plt.ylabel('success_perc')

ax_1 = fig1.add_subplot(2, 1, 2)
ax_1 .plot(X, Y2)
plt.xlabel('time')
plt.ylabel('avg_end2end_delay')

plt.savefig(f'{test_result_dir}/SA_weighted-f05d05.png', bbox_inches="tight", pad_inches=2, edgecolor='black',
            transparent=None)

