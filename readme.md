Google Maps Restaurant Type Classification
==========================================

###  

Description:
------------

A large percentage of restaurants in Google Maps are either labelled null or
generic as seen in Google Maps/Places which need to be tagged to correct
cuisine/type of restaurant.

 

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

 

### Cleaning Restaurant Types:

The restaurant types are combined for similar cuisines as seen below

```SPARQL
df_attrs_clean = df_rest_attrs.withColumn('rest_type', 
              F.when(F.lower(F.col('rest_type')).rlike('taco|mexican|burrito|mex'), F.lit('mexican'))\
              .when(F.lower(F.col('rest_type')).rlike('bar|pub|brewery|beer|gastropub|brasserie|bistro'), F.lit('bar'))\
              .when(F.lower(F.col('rest_type')).rlike('pizza'), F.lit('pizza'))\
              .when(F.lower(F.col('rest_type')).rlike('irish|fish & chips|fish and chips'), F.lit('irish'))\
              .when(F.lower(F.col('rest_type')).rlike('indian|pakistan|nepalese'), F.lit('indian'))\
              .when(F.lower(F.col('rest_type')).rlike('chinese|dim|sichuan|noodle|mandarin|shang|hong|\
                                                      |hot pot|餐馆|cantonese|dumpling'), F.lit('chinese'))\
              .when(F.lower(F.col('rest_type')).rlike('american|us|hot dog|diner|southern|cajun'), F.lit('american'))\
              .when(F.lower(F.col('rest_type')).rlike('ramen|japanese|izakaya'), F.lit('japanese'))\
              .when(F.lower(F.col('rest_type')).rlike('sushi'), F.lit('sushi'))\
              .when(F.lower(F.col('rest_type')).rlike('steak'), F.lit('steak'))\
              .when(F.lower(F.col('rest_type')).rlike('thai|cambodia'), F.lit('thai'))\
              .when(F.lower(F.col('rest_type')).rlike('chicken'), F.lit('fast food'))\
              .when(F.lower(F.col('rest_type')).rlike('vietnamese|pho'), F.lit('vietnamese'))\
              .when(F.lower(F.col('rest_type')).rlike('italian|pasta'), F.lit('italian'))\
                                          .when(F.lower(F.col('rest_type')).rlike('portuguese|salvadoran|peru|\
                                                    |cuban|brazilian|honduran|latin|guatemalan|ecuadorian|\
                                                      |argentinian|venezuelan|colombian|dominican|nicaraguan')\
                    , F.lit('latin american'))\
                                     
              
              .when(F.lower(F.col('rest_type')).rlike('greek|gyro|falafel|middle eastern|afghani|persian|\
                                                  |halal|kebab|mediterranean|middle eastern|lebanese'), F.lit('mediterranean'))\
              .when(F.lower(F.col('rest_type')).rlike('korean|한식당|음식점'), F.lit('korean'))\
              .when(F.lower(F.col('rest_type')).rlike('french|crêperie|crepe'), F.lit('french'))\
              .when(F.lower(F.col('rest_type')).rlike('lounge|bistro'), F.lit('bistro'))\
              .when(F.lower(F.col('rest_type')).rlike('burger|sandwich'), F.lit('burger'))\
              .when(F.lower(F.col('rest_type')).rlike('brunch|breakfast|pancake'), F.lit('brunch'))\
              .when(F.lower(F.col('rest_type')).rlike('coffee|cafe|bake|bakery|donut|bagel'), F.lit('cafe&bakery'))\
              .when(F.lower(F.col('rest_type')).rlike('southern|soul food'), F.lit('southern'))\
              .when(F.lower(F.col('rest_type')).rlike('spanish|tapas'), F.lit('spanish'))\
              .when(F.lower(F.col('rest_type')).rlike('asian'), F.lit('asian'))\
              .when(F.lower(F.col('rest_type')).rlike('european|german|polish|belgian|british|swedish|irish'),   F.lit('european'))\
              .when(F.lower(F.col('rest_type')).rlike('african'), F.lit('african'))\
              .when(F.lower(F.col('rest_type')).rlike('vegan|vegetarian'), F.lit('vegetarian'))\
              .when(F.lower(F.col('rest_type')).rlike('stand|venue|club|mall|alley|market|store|association|\
                                                      |station|juice|ice cream|center|theater|cater|court|fuel|\
                                                      |tobacco|arcade|producer|winery|yogurt|hall|school|grovery|service|\
                                                      |apartment|agency|organization|atm|estate|office|casino|\
                                                      |company|consultant|gift|deli'), F.lit('remove')) \
              .otherwise(F.lit(F.col('rest_type'))))
```

 

 

### Model:

 Sequential Model with Word Embeddings

```python
EMBED_DIM = 512
model = Sequential()
model.add(Embedding(input_dim=total_words, 
                           output_dim=EMBED_DIM, 
                           input_length=max_length))

model.add(GlobalMaxPool1D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(15, activation='softmax'))
```



### Model Results:

 ![](https://github.com/swami84/NLP_Text-Classification/blob/main/data/output/classification_heatmap_normalized.png) 

- Model Accuracy > 80%
- With American and Seafood we see lower accuracy (60-70%)
- Mexican , Thai, Chinese and India Restaurants have higher accuracy

 
