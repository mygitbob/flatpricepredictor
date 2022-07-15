import joblib

plz_steil ={
 13359: 'Gesundbrunnen', 10587: 'Charlottenburg', 12055: 'Neukölln', 10785: 'Tiergarten', 12489: 'Köpenick', 12487: 'Treptow', 12107: 'Tempelhof', 12047: 'Neukölln',
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

s_vec={
 'Biesdorf': 0,'Buch': 1,'Buckow': 2,'Charlottenburg': 3,'Dahlem': 4,
 'Französisch Buchholz': 5,'Friedenau': 6,'Friedrichsfelde': 7,'Friedrichshain': 8,
 'Gesundbrunnen': 9,'Grunewald': 10,'Heiligensee': 11,'Heinersdorf': 12,'Hellersdorf': 13,
 'Hohenschönhausen': 14,'Karlshorst': 15,'Kreuzberg': 16,'Köpenick': 17,
 'Lichtenberg': 18,'Lichterfelde': 19,'Mahlsdorf': 20,'Marzahn': 21,
 'Mitte': 22,'Moabit': 23,'Neukölln': 24,'Niederschönhausen': 25,
 'Pankow': 26,'Prenzlauer Berg': 27,'Reinickendorf': 28,'Rosenthal': 29,
 'Rudow': 30,'Rummelsburg': 31,'Schöneberg': 32,'Spandau': 33,'Steglitz': 34,
 'Tegel': 35,'Tempelhof': 36,'Tiergarten': 37,'Treptow': 38,'Wannsee': 39,
 'Wedding': 40,'Weissensee': 41,'Westend': 42,'Wilmersdorf': 43,'Zehlendorf': 44}

min_qm = 0
max_qm = 1000

min_zimmer = 1
max_zimmer = 20

qmpreis_mean = 6346.5471206388165
qm_median = 5535.185185185185



