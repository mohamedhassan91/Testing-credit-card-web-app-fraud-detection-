from flask import Flask , render_template , request
from threading import Thread

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/train")
def train_new_model():
    trainer_job = Thread(target=run)
    trainer_job.start()
    return "training started !!"

@app.route("/load_model")
def load_new_model():
    cur_model = load_model()
    return "model updated !"

@app.route("/clf", methods=['POST'])
def classify():
    # load doc
    data = request.json['data']
    cur_model = load_model()
    pred = cur_model.predict(data)
    return jsonify({
        "prediction": pred[0]
    })    

if __name__ == "__main__":
    app.run()

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
