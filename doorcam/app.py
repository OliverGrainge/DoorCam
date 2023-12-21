from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Initial DataFrame
data = {'Column1': [1, 2, 3], 'Column2': [4, 5, 6]}
df = pd.DataFrame(data)

@app.route('/')
def home():
    return render_template('display.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/update', methods=['POST'])
def update_data():
    global df
    new_data = request.json  # Assuming JSON format
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)
    return {"status": "DataFrame updated"}

if __name__ == '__main__':
    app.run(debug=True)