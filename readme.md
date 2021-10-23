Google Maps Restaurant Type Classification
==========================================

###  

Description:
------------

A large percentage of restaurants are labelled null or generic as seen in Google
Maps/Places which need to be tagged to correct cuisine/type of restaurant.

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+-----------+-------------+-------------------+
|rest_type  |cnt_per_group|perc_of_count_total|
+-----------+-------------+-------------------+
|restaurant |9992         |16.69925628812568  |
|mexican    |4073         |6.807052728336258  |
|pizza      |2862         |4.783153672599649  |
|null       |2401         |4.0127015960558206 |
|fast food  |2328         |3.89069942341439   |
|sandwich   |2235         |3.735271997994485  |
|chinese    |2145         |3.584858360491351  |
|american   |1930         |3.225536893122754  |
|italian    |1693         |2.829447647697836  |
|coffee     |1693         |2.829447647697836  |
|seafood    |995          |1.6629063257290881 |
|japanese   |970          |1.621124759755996  |
|thai       |922          |1.5409041530876577 |
|bar & grill|921          |1.539232890448734  |
|sushi      |887          |1.4824099607253278 |
|hamburger  |848          |1.4172307178073034 |
|chicken    |828          |1.3838054650288292 |
|cafe       |717          |1.1982953121082978 |
|bar        |650          |1.0863207153004095 |
|indian     |614          |1.026155260299156  |
+-----------+-------------+-------------------+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

 

To resolve this issue, we use a NLP based classification model on known
restaurant types as labels and reviews as input.

 

 

Model Results:

 

[Heatmap](https://github.com/swami84/NLP_Text-Classification/blob/main/data/output/classification_heatmap.png)
