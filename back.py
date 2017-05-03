from flask import Flask, render_template, jsonify
import numpy as np

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/get_text', methods=['GET', 'POST'])
def get_text():
    a = np.random.rand(10)
    b = np.random.rand(10)
    return jsonify({'a': a.tolist(), 'b': b.tolist()})

if __name__ == '__main__':
    app.run()