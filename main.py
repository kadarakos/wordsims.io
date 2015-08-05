import os
from bottle import route, request, static_file, run, template
from wordsims.wordsims import WordSim

@route('/')
def root():
    return static_file('test.html', root='.')

@route('/upload', method='POST')
def do_upload():
    category = request.forms.get('category')
    upload = request.files.get('upload')
    distance = 'cos' if request.forms.get('distance') == 'cosine' else 'jsd'
    name, ext = os.path.splitext(upload.filename)
    
    #if ext not in ('.png', '.jpg', '.jpeg'):
    #    return "File extension not allowed."

    save_path = "./tmp/{category}".format(category=category)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = "{path}/{file}".format(path=save_path, file=upload.filename)
    upload.save(file_path)
    print distance
    w = WordSim('csv', file_path, distance)
    global file_path
    
    try:
        html, csv = w.similarity_report(file_path+'.csv')
    except ValueError:
        return template('error.html', message="""Sorry, but the vectors do not 
                                                seem to represent
                                                a probability distribution""")
    #return "File successfully saved to '{0}'.".format(save_path)
    return template('results.html', table=html)

@route('/download', method='POST')
def do_download():
    return static_file(file_path+'.csv', root='.')



if __name__ == '__main__':
    run(host='localhost', port=8080)
    #run(host='0.0.0.0', port=8000) 
