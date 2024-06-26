from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Memuat dataset Liver Disorders yang sudah dimuat sebelumnya
df_liver = pd.read_csv('liver_disorders_cleaned.csv')  # contoh jika dataset disimpan dalam file csv

# Memilih fitur dan target
X = df_liver[['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']]
y = df_liver['selector']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

# Inisialisasi classifier KNN dengan k=3
classifier = KNeighborsClassifier(n_neighbors=3)

# Latih model pada set pelatihan
classifier.fit(X_train, y_train)

# Simpan model dan encoder
joblib.dump(classifier, 'knn_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        mcv = float(request.form['mcv'])
        alkphos = float(request.form['alkphos'])
        sgpt = float(request.form['sgpt'])
        sgot = float(request.form['sgot'])
        gammagt = float(request.form['gammagt'])
        drinks = float(request.form['drinks'])
        
        # Memuat model dan encoder
        model = joblib.load('knn_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Prediksi untuk instance baru
        new_data = [[mcv, alkphos, sgpt, sgot, gammagt, drinks]]
        prediction = model.predict(new_data)
        result = label_encoder.inverse_transform(prediction)[0]

        if result == 1:
            message = "Pasien memiliki gangguan hati."
        else:
            message = "Pasien tidak memiliki gangguan hati."

        return render_template('template.html', prediction=result, message=message)
    
    return render_template('template.html')

if __name__ == '__main__':
    app.run(debug=True)
