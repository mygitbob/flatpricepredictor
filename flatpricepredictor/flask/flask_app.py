from flask import Flask, render_template, request, redirect
import calcFlatPrice as src

app = Flask(__name__)

@app.route('/')
def startpage():
    return redirect('/search')

@app.route('/search')
def search():
    return render_template('auswahl.html', the_title='Berliner Immobilienrechner', data=src.s_vec)

@app.route('/calculate', methods=['post'])
def calc():
    try:
        groesse = int(request.form['die_groesse'])
        if groesse < 10 or groesse > 300 : raise
        zimmer = float(request.form['die_zimmer'])
        if zimmer < 1 or zimmer > 10 : raise
        steil = str(request.form['der_stadtteil'])
        if steil not in src.s_vec.keys() and steil != 'alle': raise
        ausstattung = str(request.form['die_ausstattung'])
        if ausstattung not in ['egal', 'gering', 'normal', 'luxus']: raise
    except:
        return redirect('/')
    if steil == 'alle':
        warnmeldung = 'Achtung: ohne Angabe eines Stadtteils wird die'\
        + 'Preisbestimmung weniger genau wiedergegeben als es mit der Angabe möglich wäre, Auswahl der Ausstattung wird bei dieser Suche eventuell nicht berücksichtigt'
    else:
        warnmeldung = ''

    ergebnis = src.predict_price(groesse, zimmer, steil, ausstattung)
    #return f'"{groesse}" "{zimmer}" "{steil}" "{ausstattung}"'
    return render_template('ergebnis.html', the_title='Ergebnis', die_groesse=groesse,
                           die_zimmer=zimmer, der_stadtteil=steil, die_ausstattung=ausstattung,
                           das_ergebnis=ergebnis, die_warnung=warnmeldung)


if __name__ == '__main__':
    app.run(debug=True)
