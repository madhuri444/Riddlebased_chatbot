from flask import Flask, request, jsonify,render_template
from MLcode import queryEmbedding


app = Flask(__name__)


    
@app.route('/', methods = ['GET'])
def index():
    return render_template("./index.html")
    
@app.route('/get-response', methods = ['POST'])
def chat():
    question = request.json.get('question')
    
    answer = queryEmbedding(question)
    
    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(debug=True)
