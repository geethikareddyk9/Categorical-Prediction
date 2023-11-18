# import tkinter as tk
# from tkinter import filedialog
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

# class CategoricalPredictionGUI:
#     def __init__(self, master):
#         self.master = master
#         master.title("Categorical Prediction")

#         self.train_file_label = tk.Label(master, text="Training data file: ")
#         self.train_file_label.pack()

#         self.train_file_button = tk.Button(master, text="Open Training File", command=self.open_train_file)
#         self.train_file_button.pack()

#         self.test_file_label = tk.Label(master, text="Test data file: ")
#         self.test_file_label.pack()

#         self.test_file_button = tk.Button(master, text="Open Test File", command=self.open_test_file)
#         self.test_file_button.pack()

#         self.run_button = tk.Button(master, text="Run", command=self.run_model)
#         self.run_button.pack()

#         self.rmse_label = tk.Label(master, text="")
#         self.rmse_label.pack()

#         self.train_file_path = None
#         self.test_file_path = None

#     def open_train_file(self):
#         self.train_file_path = filedialog.askopenfilename(initialdir="/", title="Select file",
#                                                           filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
#         self.train_file_label.config(text=f"Training data file: {self.train_file_path}")

#     def open_test_file(self):
#         self.test_file_path = filedialog.askopenfilename(initialdir="/", title="Select file",
#                                                          filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
#         self.test_file_label.config(text=f"Test data file: {self.test_file_path}")

#     def run_model(self):
#         if self.train_file_path is None or self.test_file_path is None:
#             self.rmse_label.config(text="Please select both training and test files.")
#             return

#         # Load data
#         train_df = pd.read_csv(self.train_file_path)
#         test_df = pd.read_csv(self.test_file_path)

#         # Encode categorical features
#         encoder = LabelEncoder()
#         for col in train_df.select_dtypes(include=['object']):
#             train_df[col] = encoder.fit_transform(train_df[col])
#             test_df[col] = encoder.transform(test_df[col])

#         # Prepare data for modeling
#         X_train = train_df.drop(['id', 'target'], axis=1)
#         y_train = train_df['target']
#         X_test = test_df.drop('id', axis=1)

#         scale = StandardScaler().fit(X_train)
#         X_train_scaled = scale.transform(X_train)

#         # Fit linear regression model
#         model = LinearRegression()
#         model.fit(X_train_scaled, y_train)

#         X_test_scaled = scale.transform(X_test)

#         # Predict target for test set
#         y_test_pred = model.predict(X_test_scaled)

#         # Calculate RMSE
#         y_train_pred = model.predict(X_train_scaled)
#         train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)

#         self.rmse_label.config(text=f"Training RMSE: {train_rmse:.4f}")

#         # save the predictions in the format specified in the sample submission file
#         submission = pd.read_csv('/content/drive/MyDrive/ML project/sample_submission.csv')
#         submission['target'] = y_test_pred
#         submission.to_csv('submission.csv', index=False)

# root = tk.Tk()
# my_gui = CategoricalPredictionGUI(root)
# root.mainloop()
import tkinter as tk
from tkinter import filedialog
import pandas as pd
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tkinter import *
from PIL import ImageTk, Image
import tkinter as tk
from tkinter.font import Font

def open_train_file():
    global train_df
    train_df = pd.read_csv(filedialog.askopenfilename(title="Select Training File", filetypes=[("CSV Files", "*.csv")]))

def open_test_file():
    global test_df
    test_df = pd.read_csv(filedialog.askopenfilename(title="Select Test File", filetypes=[("CSV Files", "*.csv")]))

def generate_submission_file():
    # Encode categorical features
    encoder = LabelEncoder()
    for col in train_df.select_dtypes(include=['object']):
        train_df[col] = encoder.fit_transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])

    # Prepare data for modeling
    X_train = train_df.drop(['id', 'target'], axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('id', axis=1)

    scale = StandardScaler().fit(X_train)
    X_train_scaled = scale.transform(X_train)

    # Fit Random Forest Regressor model
    model = RandomForestRegressor(n_estimators = 100, random_state = 0)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scale.transform(X_test)

    # Predict target for test set
    y_test_pred = model.predict(X_test_scaled)

    # Calculate RMSE
    y_train_pred = model.predict(X_train_scaled)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    print(f'Training RMSE: {train_rmse:.4f}')

    # Create submission file
    submission = pd.DataFrame({'id': test_df['id'], 'target': y_test_pred})
    submission.to_csv(filedialog.asksaveasfilename(title="Save Submission File", filetypes=[("CSV Files", "*.csv")]), index=False)

# Create the Tkinter interface
root = tk.Tk()
root.title("Categorical Prediction")
image = Image.open("E:/ML project/ml.webp")
background_image = ImageTk.PhotoImage(image)

# Create a label and add the image to it
label = Label(root, image=background_image)
label.place(x=0, y=0, relwidth=1, relheight=1)
background_image = ImageTk.PhotoImage(image)
registration_frame = tk.Frame(root, bd=2, relief=tk.GROOVE)
registration_frame.pack(padx=400, pady=100)

input_frame = tk.Frame(root, borderwidth=5, relief="ridge")
input_frame.place(relx=0.5, rely=0.15, relwidth=0.3, relheight=0.8, anchor="n")
# Create a label and add the image to it
label = Label(root, image=background_image)
label.place(x=0, y=0, relwidth=1, relheight=1)
# Create the label and button widgets
train_label = tk.Label(root, text="Training data file")
train_label.pack(pady=10)
font = Font(size=20)
train_label.config(font=font)
button_train = tk.Button(root, text="Upload Training File", command=open_train_file)
button_train.pack(pady=10)
font = Font(size=10)
button_train.config(font=font)
label = tk.Label(root, text="Test data file")
label.pack(pady=10)
font = Font(size=20)
label.config(font=font)
button_test = tk.Button(root, text="Upload Test File", command=open_test_file)
button_test.pack(pady=10)
font = Font(size=10)
button_test.config(font=font)

button_generate = tk.Button(root, text="Generate Submission File", command=generate_submission_file)
button_generate.pack(pady=10)
font = Font(size=10)
button_generate.config(font=font)

root.mainloop()

