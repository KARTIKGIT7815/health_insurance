#Required Libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd


app = Flask(__name__)


@app.route('/')
# Home Page
def index():
    #Details Filling Page
    return render_template('Page1.html')

#Predection Page
@app.route('/submit', methods=['POST'])
def f2():

    #Data Collected In the Dictionary Fromat Coming from HTML Page
    form_data = request.form.to_dict()

    #Loop is used to Check the key is Digit or String
    for key in form_data:
        if form_data[key].isdigit(): 
            form_data[key] = float(form_data[key])

    #Data Frame Created using User Input
    df = pd.DataFrame([form_data])
    print(df)

    #DF Information
    print(df.info())

    #DF How Many Rows & Columns
    print(df.shape)

    #Categorical Columns
    cat_col = list(df.select_dtypes(include=['object']).columns)
    print('Category Columns',cat_col)

    #Numerical Columns
    num_col = list(df.select_dtypes(exclude=['object']).columns)
    print('Numeric Columns',num_col)

    #Scaling Model
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    #Encoding Model
    encoder = pickle.load(open('encoder.pkl', 'rb'))

    #Trained Model
    model = pickle.load(open('model.pkl','rb'))

    #Scaling on new DataFrame
    S = df[num_col]
    S_scaled = scaler.transform(S)
    print('Scaling_Value',S_scaled)

    #Encoing on new DataFrame
    E = df[cat_col]
    E_encoded = encoder.transform(E)
    print('Encoding_Value',E_encoded)

    #Creting New Dataset
    sel_col = pd.concat([pd.DataFrame(S_scaled), pd.DataFrame(E_encoded)], axis=1)
    sel_col_1 = sel_col.to_numpy()
    print('Selected_Col',sel_col_1)

    #Model Predection
    yp = model.predict(sel_col_1)
    predicted_price = str(round(yp[0],2))
    print(predicted_price)

    #Predection Details
    return render_template('Page2.html',Health=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)