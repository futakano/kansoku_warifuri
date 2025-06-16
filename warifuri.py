import pandas as pd
from pulp import LpProblem, LpVariable, LpMaximize, LpBinary, lpSum, LpStatus

# ----- データ読み込み -----
file_path = "kaitou.csv"
df = pd.read_csv(file_path, header=None)
df.columns = ["student_id", "first", "second", "third"]

# ----- 科目名からIDへのマッピング -----
subject_columns = ["first", "second", "third"]
unique_subjects = ["測地観測", "大気物理・化学観測", "火山化学観測", "地球電磁気学観測", "海洋物理学観測", "夜間大気光観測", "地震観測"]

subject_to_id = {name: i for i, name in enumerate(unique_subjects)}
id_to_subject = {i: name for name, i in subject_to_id.items()}
subjects = list(subject_to_id.values())

# ----- 学生と希望辞書の作成 -----
students = df["student_id"].tolist()
preferences = {}
for _, row in df.iterrows():
    s_id = row["student_id"]
    prefs = []
    for col in subject_columns:
        subject = row[col]
        if pd.notna(subject):
            prefs.append(subject_to_id[str(subject).strip()])
    preferences[s_id] = prefs

# ----- スコア関数 -----
def get_score(student, subject):
    prefs = preferences.get(student, [])
    if subject in prefs:
        return 3 - prefs.index(subject)
    return 0

# ----- 最適化問題定義 -----
prob = LpProblem("StudentAssignment", LpMaximize)
x = {
    (i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary)
    for i in students
    for j in subjects
}

# ----- 目的関数 -----
prob += lpSum(get_score(i, j) * x[i, j] for i in students for j in subjects)

# 各学生は1科目に割り当て
for i in students:
    prob += lpSum(x[i, j] for j in subjects) == 1

# ----- 科目ごとの人数制限（個別に設定） -----
subject_limits = {
    "測地観測": (4, 5),
    "大気物理・化学観測": (1, 2),
    "火山化学観測": (4, 5),
    "地球電磁気学観測": (4, 5),
    "海洋物理学観測": (5, 5),
    "夜間大気光観測": (3, 3),
    "地震観測": (4, 5),
}

for j in subjects:
    name = id_to_subject[j]
    if name in subject_limits:
        lower, upper = subject_limits[name]
        total = lpSum(x[i, j] for i in students)
        prob += total >= lower
        prob += total <= upper
    else:
        # 制限が未設定の科目には任意の人数を許可（上限100などにする）
        total = lpSum(x[i, j] for i in students)
        prob += total <= 100

# ----- 解く -----
status = prob.solve()
print("Solver status:", LpStatus[status])

# ----- 結果出力 -----
assignment = {
    i: j for i in students for j in subjects if x[i, j].varValue == 1
}

# 割り当て済みの生徒ID
assigned_ids = set(assignment.keys())

# 未割り当ての生徒ID
unassigned_ids = [i for i in students if i not in assigned_ids]

# 割り当て済み + 未割り当てを統合して結果に含める
full_result_df = pd.DataFrame({
    "student_id": students,
    "assigned_subject": [
        id_to_subject[assignment[i]] if i in assignment else "未割り当て"
        for i in students
    ]
})

# 結果保存
full_result_df.to_csv("assignment_result2.csv", index=False)

# 科目ごとの人数分布も出力
print(full_result_df["assigned_subject"].value_counts())
