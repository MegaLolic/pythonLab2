import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

df_final = pd.read_csv(r"D:\\pythin\\processed_titanic.csv")

X = df_final.drop(columns=['Transported'])
y = df_final['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

y_regression = df_final['Age']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_log_reg = log_reg.predict(X_test)
print("\nСредняя квадратичная ошибка")
print(mean_squared_error(y_test, y_pred_log_reg))
print("\nКорень среднеквадратичной ошибки")
print(root_mean_squared_error(y_test, y_pred_log_reg))
print("\nСредняя абсолютная ошибка")
print(mean_absolute_error(y_test, y_pred_log_reg))

print("\nОценка модели логистической регрессии:")
print(classification_report(y_test, y_pred_log_reg))

cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
sns.heatmap(cm_log_reg, annot=True, fmt="d", cmap="Blues")
plt.title("Матрица ошибок (Логистическая регрессия)")
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X_test, y_test)

y_pred_lin_reg = lin_reg.predict(X_test)
print("\nСредняя квадратичная ошибка")
print(mean_squared_error(y_test, y_pred_lin_reg))
print("\nКорень среднеквадратичной ошибки")
print(root_mean_squared_error(y_test, y_pred_lin_reg))
print("\nСредняя абсолютная ошибка")
print(mean_absolute_error(y_test, y_pred_lin_reg))

y_pred_lin_reg_class = [1 if p > 0.5 else 0 for p in y_pred_lin_reg]
print("\nОценка модели линейной регрессии:")
print(classification_report(y_test, y_pred_lin_reg_class))

cm_lin_reg = confusion_matrix(y_test, y_pred_lin_reg_class)
sns.heatmap(cm_lin_reg, annot=True, fmt="d", cmap="Blues")
plt.title("Матрица ошибок (Линейная регрессия)")
plt.show()




