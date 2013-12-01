from flask import Flask
from flask import render_template, request
from flask.ext.assets import Environment, Bundle
from regress import model, vectorizer, load_reddit

app = Flask(__name__)
assets = Environment(app)

data, target = load_reddit('reddit.csv', debug=True)
model.fit(data, target)

# static assets
js = Bundle('js/vendor/modernizr-2.6.2-respond-1.1.0.min.js', 
            'js/vendor/jquery-1.10.1.min.js', 
            'js/vendor/bootstrap.min.js', 
            'js/main.js', filters='jsmin', output='gen/packed.js')

css = Bundle('css/bootstrap.min.css', 
             'css/main.css', filters='cssmin', output='gen/packed.css')
assets.register('js_all', js)
assets.register('css_all', css)

@app.route('/', methods=['GET'])
def index():
    title = request.args.get('title', None)
    if title:
        X = vectorizer.transform([title]).toarray()
        y = model.predict(X)
        print y
    return render_template('index.html', title=title, pred=y[0])

if __name__ == '__main__':
    app.run(debug=True)