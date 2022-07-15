import numpy as np
import pandas as pd
import joblib
import os
from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')

path_to_models = '/home/bbob75/mysite/models/'

plz_steil ={13359: 'Gesundbrunnen', 10587: 'Charlottenburg', 12055: 'Neukölln', 10785: 'Tiergarten', 12489: 'Köpenick', 12487: 'Treptow', 12107: 'Tempelhof', 12047: 'Neukölln',
 12203: 'Lichterfelde', 10555: 'Tiergarten', 10559: 'Tiergarten', 13595: 'Spandau', 10317: 'Lichtenberg', 10965: 'Kreuzberg', 12169: 'Steglitz', 12459: 'Treptow', 10437: 'Prenzlauer',
 13351: 'Wedding', 10553: 'Tiergarten', 10117: 'Mitte', 13509: 'Reinickendorf', 13407: 'Reinickendorf', 12159: 'Friedenau', 13507: 'Reinickendorf', 10707: 'Wilmersdorf', 12157: 'Steglitz',
 13587: 'Spandau', 14165: 'Zehlendorf', 12205: 'Lichterfelde', 10585: 'Charlottenburg', 10247: 'Friedrichshain', 10245: 'Friedrichshain', 13086: 'Weissensee', 10407: 'Prenzlauer',
 13349: 'Wedding', 10967: 'Neukölln', 12167: 'Steglitz', 10963: 'Kreuzberg', 12103: 'Tempelhof', 12165: 'Steglitz', 12247: 'Steglitz', 13409: 'Reinickendorf', 14059: 'Charlottenburg',
 13156: 'Niederschönhausen', 10115: 'Wedding', 10405: 'Prenzlauer', 13357: 'Gesundbrunnen', 10589: 'Charlottenburg', 10823: 'Schöneberg', 12555: 'Köpenick', 12305: 'Tempelhof', 10179: 'Mitte',
 10439: 'Prenzlauer', 13158: 'Pankow', 10709: 'Wilmersdorf', 12207: 'Steglitz', 12524: 'Köpenick', 12051: 'Neukölln', 14129: 'Zehlendorf', 10315: 'Lichtenberg', 10627: 'Charlottenburg',
 13187: 'Pankow', 10717: 'Wilmersdorf', 12101: 'Tempelhof', 12105: 'Tempelhof', 12249: 'Steglitz', 13403: 'Reinickendorf', 14197: 'Wilmersdorf', 12309: 'Tempelhof', 12683: 'Hellersdorf',
 13505: 'Reinickendorf', 14193: 'Zehlendorf', 13353: 'Wedding', 12627: 'Hellersdorf', 10243: 'Friedrichshain', 12045: 'Neukölln', 14167: 'Zehlendorf', 12587: 'Köpenick', 12557: 'Köpenick',
 10551: 'Tiergarten', 13189: 'Pankow', 10711: 'Wilmersdorf', 12351: 'Neukölln', 10719: 'Wilmersdorf', 13583: 'Spandau', 12279: 'Steglitz', 10249: 'Friedrichshain', 10825: 'Schöneberg',
 13053: 'Hohenschönhausen', 10999: 'Kreuzberg', 10827: 'Schöneberg', 12355: 'Rudow', 13591: 'Spandau', 13585: 'Spandau', 10997: 'Kreuzberg', 10777: 'Schöneberg', 10625: 'Charlottenburg',
 14195: 'Wilmersdorf', 13593: 'Spandau', 12623: 'Mahlsdorf', 13055: 'Hohenschönhausen', 10119: 'Mitte', 14169: 'Steglitz', 13059: 'Hohenschönhausen', 13347: 'Wedding', 12621: 'Hellersdorf',
 12043: 'Neukölln', 12307: 'Tempelhof', 13088: 'Weissensee', 12349: 'Neukölln', 12347: 'Neukölln', 10629: 'Charlottenburg', 10365: 'Lichtenberg', 14052: 'Charlottenburg', 10409: 'Prenzlauer',
 14199: 'Wilmersdorf', 10318: 'Karlshorst', 10969: 'Mitte', 12109: 'Tempelhof', 12357: 'Buckow', 10435: 'Mitte', 13127: 'Französisch', 10369: 'Lichtenberg', 12049: 'Neukölln', 12099: 'Tempelhof',
 13467: 'Reinickendorf', 13465: 'Reinickendorf', 14163: 'Zehlendorf', 10787: 'Schöneberg', 10623: 'Charlottenburg', 14057: 'Charlottenburg', 12359: 'Neukölln', 12527: 'Köpenick', 13503: 'Heiligensee',
 12161: 'Friedenau', 10557: 'Mitte', 12435: 'Treptow', 12685: 'Marzahn', 12559: 'Köpenick', 12619: 'Hellersdorf', 10178: 'Mitte', 13599: 'Spandau', 14050: 'Westend', 13089: 'Heinersdorf',
 10715: 'Wilmersdorf', 13581: 'Spandau', 14109: 'Wannsee', 12057: 'Treptow', 12053: 'Neukölln', 14055: 'Westend', 13629: 'Spandau', 12059: 'Neukölln', 10779: 'Wilmersdorf', 12277: 'Tempelhof',
 10961: 'Kreuzberg', 13469: 'Reinickendorf', 12437: 'Treptow', 10781: 'Schöneberg', 13405: 'Reinickendorf', 13439: 'Reinickendorf', 10319: 'Lichtenberg', 12163: 'Steglitz', 12353: 'Neukölln',
 10367: 'Lichtenberg', 10789: 'Tiergarten', 10783: 'Schöneberg', 10713: 'Wilmersdorf', 13589: 'Spandau', 13597: 'Spandau', 13125: 'Buch', 13627: 'Charlottenburg'}

s_vec={'Biesdorf': 0,
 'Buch': 1,
 'Buckow': 2,
 'Charlottenburg': 3,
 'Dahlem': 4,
 'Französisch Buchholz': 5,
 'Friedenau': 6,
 'Friedrichsfelde': 7,
 'Friedrichshain': 8,
 'Gesundbrunnen': 9,
 'Grunewald': 10,
 'Heiligensee': 11,
 'Heinersdorf': 12,
 'Hellersdorf': 13,
 'Hohenschönhausen': 14,
 'Karlshorst': 15,
 'Kreuzberg': 16,
 'Köpenick': 17,
 'Lichtenberg': 18,
 'Lichterfelde': 19,
 'Mahlsdorf': 20,
 'Marzahn': 21,
 'Mitte': 22,
 'Moabit': 23,
 'Neukölln': 24,
 'Niederschönhausen': 25,
 'Pankow': 26,
 'Prenzlauer Berg': 27,
 'Reinickendorf': 28,
 'Rosenthal': 29,
 'Rudow': 30,
 'Rummelsburg': 31,
 'Schöneberg': 32,
 'Spandau': 33,
 'Steglitz': 34,
 'Tegel': 35,
 'Tempelhof': 36,
 'Tiergarten': 37,
 'Treptow': 38,
 'Wannsee': 39,
 'Wedding': 40,
 'Weissensee': 41,
 'Westend': 42,
 'Wilmersdorf': 43,
 'Zehlendorf': 44}

s_median = {'Biesdorf': 5742.113679313853,
 'Buch': 6376.3066202090595,
 'Buckow': 6732.869910625621,
 'Charlottenburg': 6183.0940719473065,
 'Dahlem': 12675.15923566879,
 'Französisch': 4818.9655172413795,
 'Friedenau': 5664.802714171269,
 'Friedrichsfelde': 4192.168674698795,
 'Friedrichshain': 5675.0,
 'Gesundbrunnen': 5800.0,
 'Grunewald': 11930.06993006993,
 'Heiligensee': 4731.539895332999,
 'Heinersdorf': 8193.80121125757,
 'Hellersdorf': 3983.1714527027025,
 'Hohenschönhausen': 4749.736939396525,
 'Karlshorst': 4791.666666666667,
 'Kreuzberg': 5546.1233729485,
 'Köpenick': 6254.319281271596,
 'Lichtenberg': 4352.7580154065745,
 'Lichterfelde': 5671.358550798837,
 'Mahlsdorf': 6428.160919540231,
 'Marzahn': 4150.6024096385545,
 'Mitte': 8318.181818181818,
 'Moabit': 6421.894498014748,
 'Neukölln': 4436.001749588233,
 'Niederschönhausen': 5367.041198501873,
 'Pankow': 5878.170099481575,
 'Prenzlauer': 7223.684210526316,
 'Reinickendorf': 4109.589041095891,
 'Rosenthal': 5620.128524046435,
 'Rudow': 7900.000000000001,
 'Rummelsburg': 6104.23688470587,
 'Schöneberg': 6136.298076923077,
 'Spandau': 4130.434782608696,
 'Steglitz': 5000.0,
 'Tegel': 4810.3448275862065,
 'Tempelhof': 4688.152676406759,
 'Tiergarten': 4873.308997208504,
 'Treptow': 4648.961030993383,
 'Wannsee': 6991.333333333334,
 'Wedding': 4666.666666666667,
 'Weissensee': 5030.391951372878,
 'Westend': 5069.444444444444,
 'Wilmersdorf': 8580.0,
 'Zehlendorf': 5382.189844514515}

steil_count = len(s_vec)

min_qm = 0
max_qm = 1000

min_zimmer = 1
max_zimmer = 20

qmpreis_mean = 6346.5471206388165
qmpreis_median = 5535.185185185185


def get_steil(plz):
    if plz not in plz_steil:
        raise Exception('plz nicht in vorhanden')
    return plz_steil[plz]

def get_input_vector(groesse, zimmer, steil):
    my_vec = [0 if steil != list(s_vec.keys())[i] else 1 for i in range(steil_count)]
    return [groesse, zimmer] + my_vec

def load_models(path=path_to_models):
    return {f.split('_')[0] + '_' + f.split('_')[2]:joblib.load(path + f)
            for f in os.listdir(path) if f[:3] in ['gbr', 'rid', 'las']}

def predict_price(groesse, zimmer, stadtteil, ausstattung, low=4968.866161393385, high=7724.2280798842485):
        ivec = get_input_vector(groesse, zimmer, stadtteil)
        mymodels = load_models()
        if stadtteil == 'alle':
            return str(
                        int(((
                        mymodels['lasso_as'].predict([[groesse, zimmer, qmpreis_median]])[0] +
                        mymodels['rid_as'].predict([[groesse, zimmer, qmpreis_median]])[0]) / 2)[0])) + ',00 €'
        if ausstattung == 'egal':
            ivec.insert(2, s_median[stadtteil])
            return str(
                        int(((
                        mymodels['lasso_aa'].predict([ivec])[0] +
                        mymodels['rid_aa'].predict([ivec])[0]) / 2)[0])) + ',00 €'
        elif ausstattung == 'gering':
            return str(
                        int(((
                        mymodels['rid_low'].predict([ivec])[0] +
                        mymodels['lasso_low'].predict([ivec])[0]) / 2)[0])) + ',00 €'
        elif ausstattung == 'luxus':
            return str(
                        int(((
                        mymodels['rid_high'].predict([ivec])[0] +
                        mymodels['lasso_high'].predict([ivec])[0]) / 2)[0])) + ',00 €'
        return str(
                    int(((
                    mymodels['rid_med'].predict([ivec])[0] +
                    mymodels['lasso_med'].predict([ivec])[0]) / 2)[0])) + ',00 €'

if __name__ == '__main__':
    print(get_steil(10551))
    try:
        print(get_steil(0))
    except: print('Die plz haben wir nicht')
    print(get_input_vector(75,3,10551))
    print(load_models())
    print(predict_price(75,3,"Buch",'egal'))

