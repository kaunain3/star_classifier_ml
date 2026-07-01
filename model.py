





df = pd.read_csv("star_data.csv")
df["Star color"] = (
    df["Star color"]
    .str.lower()
    .str.replace("-", " ", regex=False)
    .str.strip()
)

df["Star color"] = df["Star color"].replace({"white yellow": "yellow white"})


df = df.drop_duplicates()


numeric_cols = [
    "Temperature (K)",
    "Luminosity(L/Lo)",
    "Radius(R/Ro)",
    "Absolute magnitude(Mv)"
]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


joblib.dump(scaler, "scaler.pkl")

encoded_df = pd.get_dummies(df, columns=["Star color", "Spectral Class"], drop_first= True)


x = encoded_df.drop("Star type", axis=1)
y = encoded_df["Star type"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)




svm = SVC()

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(x_train, y_train)


best_svm = grid_search.best_estimator_
y_true = y_test
y_pred = best_svm.predict(x_test)


joblib.dump(best_svm, "svm.pkl") 



