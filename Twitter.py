{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# RECOMENDADOR DE CANCIONES"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"generos.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [],
   "source": [
    "df= df.sample(1000)\n",
    "df = df.reset_index()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>index</th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>artist</th>\n",
       "      <th>seq</th>\n",
       "      <th>song</th>\n",
       "      <th>label</th>\n",
       "      <th>name</th>\n",
       "      <th>spotify_uri</th>\n",
       "      <th>genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>304</th>\n",
       "      <td>77315</td>\n",
       "      <td>74895</td>\n",
       "      <td>Lightnin' Hopkins</td>\n",
       "      <td>Troubled in mind, I'm little blue\\nBet you, I ...</td>\n",
       "      <td>Trouble in Mind [#]</td>\n",
       "      <td>0.675</td>\n",
       "      <td>Lightnin' Hopkins</td>\n",
       "      <td>spotify:artist:6EZzVXM2uDRPmnHWq9yPDE</td>\n",
       "      <td>['acoustic blues', 'blues', 'country blues', '...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>423</th>\n",
       "      <td>21780</td>\n",
       "      <td>21445</td>\n",
       "      <td>Trisha Yearwood</td>\n",
       "      <td>Mama told her baby girl take it real slow\\nGir...</td>\n",
       "      <td>Walkaway Joe</td>\n",
       "      <td>0.428</td>\n",
       "      <td>Trisha Yearwood</td>\n",
       "      <td>spotify:artist:3XlIhgydjvC4EniPFZT20j</td>\n",
       "      <td>['contemporary country', 'country', 'country d...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>146696</td>\n",
       "      <td>150860</td>\n",
       "      <td>Blood Orange</td>\n",
       "      <td>It was the last weekend of September, I stayed...</td>\n",
       "      <td>Chosen</td>\n",
       "      <td>0.190</td>\n",
       "      <td>Blood Orange</td>\n",
       "      <td>spotify:artist:6LEeAFiJF8OuPx747e1wxR</td>\n",
       "      <td>['alternative r&amp;b', 'art pop', 'chillwave', 'e...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      index  Unnamed: 0             artist  \\\n",
       "304   77315       74895  Lightnin' Hopkins   \n",
       "423   21780       21445    Trisha Yearwood   \n",
       "7    146696      150860       Blood Orange   \n",
       "\n",
       "                                                   seq                 song  \\\n",
       "304  Troubled in mind, I'm little blue\\nBet you, I ...  Trouble in Mind [#]   \n",
       "423  Mama told her baby girl take it real slow\\nGir...         Walkaway Joe   \n",
       "7    It was the last weekend of September, I stayed...               Chosen   \n",
       "\n",
       "     label               name                            spotify_uri  \\\n",
       "304  0.675  Lightnin' Hopkins  spotify:artist:6EZzVXM2uDRPmnHWq9yPDE   \n",
       "423  0.428    Trisha Yearwood  spotify:artist:3XlIhgydjvC4EniPFZT20j   \n",
       "7    0.190       Blood Orange  spotify:artist:6LEeAFiJF8OuPx747e1wxR   \n",
       "\n",
       "                                                genres  \n",
       "304  ['acoustic blues', 'blues', 'country blues', '...  \n",
       "423  ['contemporary country', 'country', 'country d...  \n",
       "7    ['alternative r&b', 'art pop', 'chillwave', 'e...  "
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.sample(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Letras: (1000, 9)\n"
     ]
    }
   ],
   "source": [
    "print(\"Letras:\",df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1000 entries, 0 to 999\n",
      "Data columns (total 9 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   index        1000 non-null   int64  \n",
      " 1   Unnamed: 0   1000 non-null   int64  \n",
      " 2   artist       1000 non-null   object \n",
      " 3   seq          1000 non-null   object \n",
      " 4   song         1000 non-null   object \n",
      " 5   label        1000 non-null   float64\n",
      " 6   name         1000 non-null   object \n",
      " 7   spotify_uri  1000 non-null   object \n",
      " 8   genres       1000 non-null   object \n",
      "dtypes: float64(1), int64(2), object(6)\n",
      "memory usage: 70.4+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.columns=['id','id2','artist','letra','cancion','label','artista2','uri','genero']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.sum(df.cancion.isnull())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.sum(df.letra.isnull())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['tema']=(df['cancion']+\"(\"+df['artist']+\")\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['id', 'id2', 'artist', 'letra', 'cancion', 'label', 'artista2', 'uri',\n",
       "       'genero', 'tema'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Content Based Recommendation System¶\n",
    "Now lets make a recommendations based on the movie’s plot summaries given in the overview column. So if our user gives us a movie title, our goal is to recommend movies that share similar plot summaries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    Get on the revolution\\r\\nGet on the revolution...\n",
       "Name: letra, dtype: object"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(1)['letra']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "\n",
    "tfv = TfidfVectorizer(min_df=3,  max_features=None, \n",
    "            strip_accents='unicode', analyzer='word',token_pattern=r'\\w{1,}',\n",
    "            ngram_range=(1, 3),\n",
    "            stop_words = 'english')\n",
    "\n",
    "# Filling NaNs with empty string\n",
    "df['letra'] = df['letra'].fillna('')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fitting the TF-IDF on the 'overview' text\n",
    "tfv_matrix = tfv.fit_transform(df['letra'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [],
   "source": [
    "tfv_matrix;"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1000, 5512)"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tfv_matrix.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics.pairwise import sigmoid_kernel\n",
    "\n",
    "# Compute the sigmoid kernel\n",
    "sig = sigmoid_kernel(tfv_matrix, tfv_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [],
   "source": [
    "sig[0];"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reverse mapping of indices and movie titles\n",
    "indices = pd.Series(df.index, index=df['tema']).drop_duplicates()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tema\n",
       "Revolution(Sophie Ellis-Bextor)                      0\n",
       "Puttin' Memories Away(Gary Allan)                    1\n",
       "Goodbye, So Long(Ike & Tina Turner)                  2\n",
       "Raven in the Snow(Bill Miller)                       3\n",
       "He Thinks He'll Keep Her(The Nashville Guitars)      4\n",
       "                                                  ... \n",
       "Put Yo Hood Up(Lil Jon)                            995\n",
       "Welcome to Dying(Blind Guardian)                   996\n",
       "Angel of Death(Thin Lizzy)                         997\n",
       "For Baltimore(All Time Low)                        998\n",
       "Fight the Fire(Primal Fear)                        999\n",
       "Length: 1000, dtype: int64"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "indices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'Nowhere to Run(Santana)'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_value\u001b[1;34m(self, series, key)\u001b[0m\n\u001b[0;32m   4410\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4411\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mlibindex\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_value_at\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ms\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4412\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mIndexError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.get_value_at\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.get_value_at\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\util.pxd\u001b[0m in \u001b[0;36mpandas._libs.util.get_value_at\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\util.pxd\u001b[0m in \u001b[0;36mpandas._libs.util.validate_indexer\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: 'str' object cannot be interpreted as an integer",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-102-fccfe71964b8>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mindices\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'Nowhere to Run(Santana)'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\series.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    869\u001b[0m         \u001b[0mkey\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcom\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply_if_callable\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    870\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 871\u001b[1;33m             \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_value\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    872\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    873\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mis_scalar\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_value\u001b[1;34m(self, series, key)\u001b[0m\n\u001b[0;32m   4417\u001b[0m                     \u001b[1;32mraise\u001b[0m \u001b[0mInvalidIndexError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4418\u001b[0m                 \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4419\u001b[1;33m                     \u001b[1;32mraise\u001b[0m \u001b[0me1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4420\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4421\u001b[0m                 \u001b[1;32mraise\u001b[0m \u001b[0me1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_value\u001b[1;34m(self, series, key)\u001b[0m\n\u001b[0;32m   4403\u001b[0m         \u001b[0mk\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_convert_scalar_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mk\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkind\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"getitem\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4404\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4405\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_value\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ms\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mk\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtz\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mgetattr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mseries\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"tz\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4406\u001b[0m         \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4407\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m0\u001b[0m \u001b[1;32mand\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mholds_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mis_boolean\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_value\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_value\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine._get_loc_duplicates\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine._maybe_get_bool_indexer\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'Nowhere to Run(Santana)'"
     ]
    }
   ],
   "source": [
    "indices['Nowhere to Run(Santana)']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [
    {
     "ename": "IndexError",
     "evalue": "index 9995 is out of bounds for axis 0 with size 1000",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mIndexError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-103-15b9e4d76f8c>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0msig\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m9995\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mIndexError\u001b[0m: index 9995 is out of bounds for axis 0 with size 1000"
     ]
    }
   ],
   "source": [
    "sig[9995]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'Nowhere to Run(Santana)'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_value\u001b[1;34m(self, series, key)\u001b[0m\n\u001b[0;32m   4410\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4411\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mlibindex\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_value_at\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ms\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4412\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mIndexError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.get_value_at\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.get_value_at\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\util.pxd\u001b[0m in \u001b[0;36mpandas._libs.util.get_value_at\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\util.pxd\u001b[0m in \u001b[0;36mpandas._libs.util.validate_indexer\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: 'str' object cannot be interpreted as an integer",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-104-566504db23c0>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mlist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0menumerate\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msig\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mindices\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'Nowhere to Run(Santana)'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\series.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    869\u001b[0m         \u001b[0mkey\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcom\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply_if_callable\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    870\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 871\u001b[1;33m             \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_value\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    872\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    873\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mis_scalar\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_value\u001b[1;34m(self, series, key)\u001b[0m\n\u001b[0;32m   4417\u001b[0m                     \u001b[1;32mraise\u001b[0m \u001b[0mInvalidIndexError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4418\u001b[0m                 \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4419\u001b[1;33m                     \u001b[1;32mraise\u001b[0m \u001b[0me1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4420\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4421\u001b[0m                 \u001b[1;32mraise\u001b[0m \u001b[0me1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_value\u001b[1;34m(self, series, key)\u001b[0m\n\u001b[0;32m   4403\u001b[0m         \u001b[0mk\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_convert_scalar_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mk\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkind\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"getitem\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4404\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4405\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_value\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ms\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mk\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtz\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mgetattr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mseries\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"tz\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4406\u001b[0m         \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4407\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m0\u001b[0m \u001b[1;32mand\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mholds_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mis_boolean\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_value\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_value\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine._get_loc_duplicates\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine._maybe_get_bool_indexer\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'Nowhere to Run(Santana)'"
     ]
    }
   ],
   "source": [
    "list(enumerate(sig[indices['Nowhere to Run(Santana)']]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [],
   "source": [
    "def give_rec(title, sig=sig):\n",
    "    # Get the index corresponding to original_title\n",
    "    idx = indices[title]\n",
    "\n",
    "    # Get the pairwsie similarity scores \n",
    "    sig_scores = list(enumerate(sig[idx]))\n",
    "\n",
    "    # Sort the movies \n",
    "    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)\n",
    "\n",
    "    # Scores of the 10 most similar movies\n",
    "    sig_scores = sig_scores[1:11]\n",
    "\n",
    "    # Movie indices\n",
    "    tema_indices = [i[0] for i in sig_scores]\n",
    "\n",
    "    # Top 10 most similar movies\n",
    "    return df['tema'].iloc[tema_indices]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'I Was the One [*](Elvis Presley)'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_value\u001b[1;34m(self, series, key)\u001b[0m\n\u001b[0;32m   4410\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4411\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mlibindex\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_value_at\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ms\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4412\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mIndexError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.get_value_at\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.get_value_at\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\util.pxd\u001b[0m in \u001b[0;36mpandas._libs.util.get_value_at\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\util.pxd\u001b[0m in \u001b[0;36mpandas._libs.util.validate_indexer\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: 'str' object cannot be interpreted as an integer",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-106-f210f325ccf8>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Testing our content-based recommendation system with the seminal film Spy Kids\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mgive_rec\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'I Was the One [*](Elvis Presley)'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-105-76507db3474e>\u001b[0m in \u001b[0;36mgive_rec\u001b[1;34m(title, sig)\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mgive_rec\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtitle\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0msig\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0msig\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m     \u001b[1;31m# Get the index corresponding to original_title\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m     \u001b[0midx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mindices\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mtitle\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m     \u001b[1;31m# Get the pairwsie similarity scores\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\series.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    869\u001b[0m         \u001b[0mkey\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcom\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply_if_callable\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    870\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 871\u001b[1;33m             \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_value\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    872\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    873\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mis_scalar\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_value\u001b[1;34m(self, series, key)\u001b[0m\n\u001b[0;32m   4417\u001b[0m                     \u001b[1;32mraise\u001b[0m \u001b[0mInvalidIndexError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4418\u001b[0m                 \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4419\u001b[1;33m                     \u001b[1;32mraise\u001b[0m \u001b[0me1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4420\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4421\u001b[0m                 \u001b[1;32mraise\u001b[0m \u001b[0me1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Continuum\\miniconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_value\u001b[1;34m(self, series, key)\u001b[0m\n\u001b[0;32m   4403\u001b[0m         \u001b[0mk\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_convert_scalar_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mk\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkind\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"getitem\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4404\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4405\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_value\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ms\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mk\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtz\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mgetattr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mseries\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"tz\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4406\u001b[0m         \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4407\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m0\u001b[0m \u001b[1;32mand\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mholds_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mis_boolean\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_value\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_value\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine._get_loc_duplicates\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine._maybe_get_bool_indexer\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'I Was the One [*](Elvis Presley)'"
     ]
    }
   ],
   "source": [
    "# Testing our content-based recommendation system with the seminal film Spy Kids\n",
    "give_rec('I Was the One [*](Elvis Presley)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import TruncatedSVD\n",
    "from sklearn.preprocessing import Normalizer\n",
    "from sklearn.pipeline import make_pipeline\n",
    "\n",
    "X=tfv_matrix\n",
    "\n",
    "svd = TruncatedSVD(30,random_state = 1) \n",
    "normalizer = Normalizer(copy=False)\n",
    "lsa = make_pipeline(svd, normalizer)\n",
    "X = lsa.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.patches as mpatches\n",
    "plt.figure()\n",
    "plt.scatter(X[:, 0], X[:, 1])\n",
    "plt.xlabel('Dimensión 1')\n",
    "plt.ylabel('Dimensión 2')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
