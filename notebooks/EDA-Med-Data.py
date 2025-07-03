{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "84187006-ad1d-4c50-8ca5-48188c2ff28d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import warnings\n",
    "\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b3a2ba56-826a-4a18-84c2-b2c060bf2399",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/Users/faes/ds-bootcamp/version1 ds-capstone-project-template/notebooks\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "print(os.getcwd())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "eaac71be-889b-44b2-98e8-e95fe66c7508",
   "metadata": {},
   "outputs": [],
   "source": [
    "# importing raw data\n",
    "df_med = pd.read_csv('/Users/faes/ds-bootcamp/version1 ds-capstone-project-template/notebooks/medquad.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c5bdf472-ad53-45f8-ae8c-51107d62c601",
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
       "      <th>question</th>\n",
       "      <th>answer</th>\n",
       "      <th>source</th>\n",
       "      <th>focus_area</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>What is (are) Glaucoma ?</td>\n",
       "      <td>Glaucoma is a group of diseases that can damag...</td>\n",
       "      <td>NIHSeniorHealth</td>\n",
       "      <td>Glaucoma</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>What causes Glaucoma ?</td>\n",
       "      <td>Nearly 2.7 million people have glaucoma, a lea...</td>\n",
       "      <td>NIHSeniorHealth</td>\n",
       "      <td>Glaucoma</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>What are the symptoms of Glaucoma ?</td>\n",
       "      <td>Symptoms of Glaucoma  Glaucoma can develop in ...</td>\n",
       "      <td>NIHSeniorHealth</td>\n",
       "      <td>Glaucoma</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>What are the treatments for Glaucoma ?</td>\n",
       "      <td>Although open-angle glaucoma cannot be cured, ...</td>\n",
       "      <td>NIHSeniorHealth</td>\n",
       "      <td>Glaucoma</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>What is (are) Glaucoma ?</td>\n",
       "      <td>Glaucoma is a group of diseases that can damag...</td>\n",
       "      <td>NIHSeniorHealth</td>\n",
       "      <td>Glaucoma</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>What is (are) Glaucoma ?</td>\n",
       "      <td>The optic nerve is a bundle of more than 1 mil...</td>\n",
       "      <td>NIHSeniorHealth</td>\n",
       "      <td>Glaucoma</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>What is (are) Glaucoma ?</td>\n",
       "      <td>Open-angle glaucoma is the most common form of...</td>\n",
       "      <td>NIHSeniorHealth</td>\n",
       "      <td>Glaucoma</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                 question  \\\n",
       "0                What is (are) Glaucoma ?   \n",
       "1                  What causes Glaucoma ?   \n",
       "2     What are the symptoms of Glaucoma ?   \n",
       "3  What are the treatments for Glaucoma ?   \n",
       "4                What is (are) Glaucoma ?   \n",
       "5                What is (are) Glaucoma ?   \n",
       "6                What is (are) Glaucoma ?   \n",
       "\n",
       "                                              answer           source  \\\n",
       "0  Glaucoma is a group of diseases that can damag...  NIHSeniorHealth   \n",
       "1  Nearly 2.7 million people have glaucoma, a lea...  NIHSeniorHealth   \n",
       "2  Symptoms of Glaucoma  Glaucoma can develop in ...  NIHSeniorHealth   \n",
       "3  Although open-angle glaucoma cannot be cured, ...  NIHSeniorHealth   \n",
       "4  Glaucoma is a group of diseases that can damag...  NIHSeniorHealth   \n",
       "5  The optic nerve is a bundle of more than 1 mil...  NIHSeniorHealth   \n",
       "6  Open-angle glaucoma is the most common form of...  NIHSeniorHealth   \n",
       "\n",
       "  focus_area  \n",
       "0   Glaucoma  \n",
       "1   Glaucoma  \n",
       "2   Glaucoma  \n",
       "3   Glaucoma  \n",
       "4   Glaucoma  \n",
       "5   Glaucoma  \n",
       "6   Glaucoma  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_med.head(7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c88e369c-8bc9-4e35-8d24-e787e7e3b835",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method DataFrame.info of                                                 question  \\\n",
       "0                               What is (are) Glaucoma ?   \n",
       "1                                 What causes Glaucoma ?   \n",
       "2                    What are the symptoms of Glaucoma ?   \n",
       "3                 What are the treatments for Glaucoma ?   \n",
       "4                               What is (are) Glaucoma ?   \n",
       "...                                                  ...   \n",
       "16407  What is (are) Diabetic Neuropathies: The Nerve...   \n",
       "16408  How to prevent Diabetic Neuropathies: The Nerv...   \n",
       "16409  How to diagnose Diabetic Neuropathies: The Ner...   \n",
       "16410  What are the treatments for Diabetic Neuropath...   \n",
       "16411  What to do for Diabetic Neuropathies: The Nerv...   \n",
       "\n",
       "                                                  answer           source  \\\n",
       "0      Glaucoma is a group of diseases that can damag...  NIHSeniorHealth   \n",
       "1      Nearly 2.7 million people have glaucoma, a lea...  NIHSeniorHealth   \n",
       "2      Symptoms of Glaucoma  Glaucoma can develop in ...  NIHSeniorHealth   \n",
       "3      Although open-angle glaucoma cannot be cured, ...  NIHSeniorHealth   \n",
       "4      Glaucoma is a group of diseases that can damag...  NIHSeniorHealth   \n",
       "...                                                  ...              ...   \n",
       "16407  Focal neuropathy appears suddenly and affects ...            NIDDK   \n",
       "16408  The best way to prevent neuropathy is to keep ...            NIDDK   \n",
       "16409  Doctors diagnose neuropathy on the basis of sy...            NIDDK   \n",
       "16410  The first treatment step is to bring blood glu...            NIDDK   \n",
       "16411  - Diabetic neuropathies are nerve disorders ca...            NIDDK   \n",
       "\n",
       "                                              focus_area  \n",
       "0                                               Glaucoma  \n",
       "1                                               Glaucoma  \n",
       "2                                               Glaucoma  \n",
       "3                                               Glaucoma  \n",
       "4                                               Glaucoma  \n",
       "...                                                  ...  \n",
       "16407  Diabetic Neuropathies: The Nerve Damage of Dia...  \n",
       "16408  Diabetic Neuropathies: The Nerve Damage of Dia...  \n",
       "16409  Diabetic Neuropathies: The Nerve Damage of Dia...  \n",
       "16410  Diabetic Neuropathies: The Nerve Damage of Dia...  \n",
       "16411  Diabetic Neuropathies: The Nerve Damage of Dia...  \n",
       "\n",
       "[16412 rows x 4 columns]>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_med.info"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0a8fe918-09b7-417d-8b08-c595f74e7281",
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
       "      <th>question</th>\n",
       "      <th>answer</th>\n",
       "      <th>source</th>\n",
       "      <th>focus_area</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>16412</td>\n",
       "      <td>16407</td>\n",
       "      <td>16412</td>\n",
       "      <td>16398</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>14984</td>\n",
       "      <td>15817</td>\n",
       "      <td>9</td>\n",
       "      <td>5126</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>What causes Causes of Diabetes ?</td>\n",
       "      <td>This condition is inherited in an autosomal re...</td>\n",
       "      <td>GHR</td>\n",
       "      <td>Breast Cancer</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>20</td>\n",
       "      <td>348</td>\n",
       "      <td>5430</td>\n",
       "      <td>53</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                question  \\\n",
       "count                              16412   \n",
       "unique                             14984   \n",
       "top     What causes Causes of Diabetes ?   \n",
       "freq                                  20   \n",
       "\n",
       "                                                   answer source  \\\n",
       "count                                               16407  16412   \n",
       "unique                                              15817      9   \n",
       "top     This condition is inherited in an autosomal re...    GHR   \n",
       "freq                                                  348   5430   \n",
       "\n",
       "           focus_area  \n",
       "count           16398  \n",
       "unique           5126  \n",
       "top     Breast Cancer  \n",
       "freq               53  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_med.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "92c8f831-3e75-412c-8c79-156debe46acd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(16412, 4)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_med.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "88cc665c-c2a3-4e26-bfdf-6e42811c46d0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False    16364\n",
       "True        48\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check how many duplicated rows exist in the data frame\n",
    "df_med.duplicated().value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8dd134c5-86f2-48d0-8392-6d1f0ee96ed9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# # remove duplicates\n",
    "# df_med = df_med.drop_duplicates()\n",
    "# # reset index inplace\n",
    "# df_med.reset_index(inplace=True, drop=True)\n",
    "# df_med.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f359fd8a-0990-42f5-a62a-16d2401da64a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "question      object\n",
       "answer        object\n",
       "source        object\n",
       "focus_area    object\n",
       "dtype: object"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check data types in data frame\n",
    "df_med.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "73197637-25fd-4e90-9804-6b3d601a87fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5126"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# display number of distinct elements\n",
    "df_med.focus_area.nunique()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "019ddc46-a4d3-4b95-ae30-6467d3d828c8",
   "metadata": {},
   "source": [
    "## Missing Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5036b54e-c869-4ef7-ae25-ef3a00034647",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import missingno\n",
    "import missingno as msno"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "ddb978bb-58da-4785-b568-f545739647e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "question       0\n",
       "answer         5\n",
       "source         0\n",
       "focus_area    14\n",
       "dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# display number of missing values per column\n",
    "df_med.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "fe67df3d-8f2d-42ea-93d5-ca8c17eeee28",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df_med.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "75b2276c-150d-4c1a-9c0c-745c4b18166f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_cleaned = df.dropna(subset=['answer'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "0769b3ad-1291-4426-bcd0-96dd83f927ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_cleaned = df_cleaned.dropna(subset=['focus_area']) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "c7aafc15-c751-4116-983a-df4332af73b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(16393, 4)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_cleaned.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "f4e2cfa0-8663-48da-bab1-50a4b8ad667e",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df_cleaned"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f61b43fd-a4e1-4a39-a3fa-f0cb925de3fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(16393, 4)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "27e63b1b-6d4b-4d9c-86d7-dc0ac6b50f2d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "focus_area\n",
       "Breast Cancer                                 53\n",
       "Prostate Cancer                               43\n",
       "Stroke                                        35\n",
       "Skin Cancer                                   34\n",
       "Alzheimer's Disease                           30\n",
       "                                              ..\n",
       "Spondylometaphyseal dysplasia type A4          1\n",
       "Hodgkin lymphoma                               1\n",
       "Muscular dystrophy white matter spongiosis     1\n",
       "Microphthalmia syndromic 9                     1\n",
       "Isolated growth hormone deficiency type 3      1\n",
       "Name: count, Length: 5125, dtype: int64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['focus_area'].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "abc1fcaf-017f-4adb-a8c2-7b35216b8d16",
   "metadata": {},
   "outputs": [],
   "source": [
    "# msno.matrix(df_med)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "05558678-a8df-449a-9d7e-5b8e83a8cfea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# df_med['focus_area'].value_counts().plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "df730da3-a3d5-41ea-98c1-5bfee0372ead",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "source\n",
       "GHR                  5430\n",
       "GARD                 5389\n",
       "NIDDK                1192\n",
       "NINDS                1088\n",
       "MPlusHealthTopics     981\n",
       "NIHSeniorHealth       769\n",
       "CancerGov             729\n",
       "NHLBI                 559\n",
       "CDC                   256\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['source'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "50bc7bd8-e069-43e4-95bc-6a3445fd4695",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='source'>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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MaPUq1dPtm3bFmefAwcOSIsWLSRr1qySPXt2adeunRw9ejTOPt9//73ceuutkiFDBilcuLAMGjToUs8RAACk9ABz7NgxqVSpkrz11lsJbtegMWLECBk9erR88803kjlzZmnQoIGcOHEiuI+Gl82bN8vChQtlzpw5LhQ98cQTwe2xsbFSv359KVq0qKxbt07++9//yssvvyzvvvvupZ4nAABIyTUwDRs2dJeEaOvLsGHDpHfv3tKkSRP32AcffCD58uVzLTUPPfSQ/PjjjzJv3jxZs2aN3HTTTW6fN998U+666y554403XMvORx99JKdOnZJx48ZJ+vTppVy5crJhwwYZMmRInKADAABSpitaA7Njxw7Zu3ev6zbyZMuWTapXry4rV6509/Vau4288KJ0/9SpU7sWG2+ff//73y68eLQVZ+vWrXLw4MEE/+2TJ0+6lpvQCwAA8KcrGmA0vChtcQml971tep03b94429OmTSs5c+aMs09CzxH6b8Q3YMAAF5a8i9bNAAAAf/LNMOpevXpJ165dg/e1BYYQA0Q+hogDCHsLTP78+d11TExMnMf1vrdNr/ft2xdn+5kzZ9zIpNB9EnqO0H8jvqioKDeqKfQCAAD86YoGmOLFi7uAsWjRojgtIVrbUrNmTXdfrw8dOuRGF3kWL14s586dc7Uy3j46Mun06dPBfXTE0g033CA5cuS4kocMAABSQoDR+Vp0RJBevMJdvb1r1y43L0znzp3lP//5j8yaNUs2btworVq1ciOL7r33Xrd/mTJl5M4775THH39cVq9eLcuXL5eOHTu6EUq6n3rkkUdcAa/OD6PDrSdPnizDhw+P00UEAABSriTXwKxdu1Zq164dvO+FitatW8v48eOle/fubq4YHe6sLS233HKLGzatE9J5dJi0hpa6deu60UfNmjVzc8d4tAh3wYIFEh0dLVWrVpXcuXO7yfEYQg0AAC4pwNx+++1uvpfEaCtMv3793CUxOuJo4sSJF/x3KlasKF999RWvEgAAOA9rIQEAAHMIMAAAwBwCDAAAMIcAAwAAzCHAAAAAcwgwAADAHAIMAAAwhwADAADM8c1q1AAQaVhpG7h6aIEBAADmEGAAAIA5dCEBAJKM7jGEGy0wAADAHAIMAAAwhwADAADMIcAAAABzCDAAAMAcAgwAADCHAAMAAMwhwAAAAHMIMAAA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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['source'].value_counts().plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "e0a18991-5562-4bed-a3a3-eac1a573151d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       question_len    answer_len\n",
      "count  16393.000000  16393.000000\n",
      "mean       8.213811    200.977673\n",
      "std        2.380932    248.189620\n",
      "min        3.000000      1.000000\n",
      "25%        6.000000     71.000000\n",
      "50%        8.000000    138.000000\n",
      "75%       10.000000    252.000000\n",
      "max       27.000000   4281.000000\n"
     ]
    }
   ],
   "source": [
    "df['question_len'] = df['question'].fillna('').apply(lambda x: len(x.split()))\n",
    "df['answer_len'] = df['answer'].fillna('').apply(lambda x: len(x.split()))\n",
    "\n",
    "print(df[['question_len', 'answer_len']].describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "86c34f03-50e7-4753-aa45-a0997eba57a9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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1qlTJ9uxY0dMmePHj9uAAQOsSpUqVrZsWevSpYvt27cvpsyuXbusQ4cOVqZMGbee4cOH26lTp2LKrFy50po1a2YlS5a0OnXq2KxZsy5mPwEAQFENMKtWrXLh5L333rNly5bZyZMnrV27dnb06NFImaFDh9qCBQts7ty5rvyePXusc+fOkeWnT5924eXEiRO2evVqe+GFF1w4GTNmTKTMzp07XZnWrVvbxo0bbciQIda3b197880382u/AQCAx5KCIAjy+uIDBw64GhQFlZtuuskOHTpkl19+ub300kvWtWtXV2b79u1Wv359W7NmjbVs2dIWL15st99+uws2qamprsz06dNt5MiRbn0lSpRwPy9atMi2bNkSea+77rrLDh48aEuWLMnRtmVkZFiFChXcNpUvX97y1YLBVmA6Tiq49wIAoJDl9Px9UX1gtHKpXLmyu9+wYYOrlWnbtm2kTL169axWrVouwIjuGzVqFAkvkp6e7jZ469atkTLR6wjLhOvITmZmpltH9A0AACSmPAeYM2fOuKad66+/3ho2bOie27t3r6tBqVixYkxZhRUtC8tEh5dwebjsfGUUSo4dO3bO/jlKbOGtZs2aed01AACQqAFGfWHUxDN79myLB6NGjXI1QuFt9+7dhb1JAADgEknJy4sGDhxoCxcutLfffttq1KgReT4tLc11zlVflehaGI1C0rKwzNq1a2PWF45Sii6TdeSSHqstrHTp0tluk0Yr6QYAABJfrmpg1N9X4WX+/Pm2YsUKq127dszy5s2bW/HixW358uWR5zTMWsOmW7Vq5R7rfvPmzbZ///5IGY1oUjhp0KBBpEz0OsIy4ToAAEDRlpLbZiONMHrttdfcXDBhnxX1OVHNiO779Oljw4YNcx17FUoGDRrkgodGIImGXSuo9OjRw8aPH+/WMXr0aLfusAalf//+NnnyZBsxYoT17t3bhaVXXnnFjUwCAADIVQ3MtGnTXP+S73//+3bFFVdEbnPmzImUmThxohsmrQnsNLRazUHz5s2LLC9WrJhrftK9gs1PfvIT69mzpz322GORMqrZUVhRrUuTJk3sqaeespkzZ7qRSAAAABc1D0w8Yx4YAAD8UyDzwAAAAHgzCgkFKCe1PdTSAACKGGpgAACAdwgwAADAOwQYAADgHQIMAADwDgEGAAB4hwADAAC8Q4ABAADeIcAA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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.hist(df['question_len'], bins=50, alpha=0.6, label='Question length')\n",
    "plt.hist(df['answer_len'], bins=50, alpha=0.6, label='Answer length')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "33b4aee9-677d-4f9a-b8e9-4a8396b35b57",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='focus_area'>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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NI9qkCRO9/gmFZ5HKjwH4XlERyo8h1R5HqnwHITncfjlUNZpNvxcAACATNHbBQAAOEXwAQAAnCL4AAAAThF8AAAAp1I++Pjrr7/MlClTzKZNm4zPdu/ebc466yyzZMmSZO9K2tNrUblyZfPdd98le1fSnl6La665xixfvtz4LFWOU5k9r3nz5pnff//d+Gjp0qXmo48+Mn/++af9mvUZh0/KBx+qtnrOOed4++YP5c6d2yxYsCDZu4H//7XYsWOHSRVffPGFufzyy20RwJ9//tlue/XVV82XX35pfHgtxo4da3yXKsepXr16meHDh8cCj2bNmplTTjnFlvSePHmy8cWvv/5qm6Eef/zx5txzzzVr166127t06WL69Oljol5CfcsB3pIp5YMPqVWrlvnxxx+N73SCCD/Yvlu2bJnp16+fufTSS82GDRvstg8//NB8++23xgfdunUzDz/8sNmzZ4/xmU7cLVu2NPnz5zdff/11rMO0uks/9NBDxgcXXniheffdd43vUuE49fbbb5uTTjrJ3h8/frwdkfr+++/NLbfcYu666y7jC+1vrly5zKpVq0yBAgUy1LBSzaooK1asmClevPh+b+H3JFNa1PnQm0UN7O6//35b+r1gwYIZHvelVn+PHj3MK6+8YqpWrZrp83j88ceNDz7//HPTunVr07hxYzvUrOmL4447zgwaNMjMnj3bHsCi7qKLLrItAwoVKmROPPHEfV6Ld955x/jg5JNPtgfaK6+80hQuXNjMnz/fvhYKRPQaqUhg1D3wwAPmscces9OSmX0ubr75ZuODVDhO5cuXz05VHHPMMea6666zJ25VuFYQoqAk2VfbB0p9yDTdon2O/1woOKxdu7bZtm2bifLx9UBpZCpZkt7V1gUNm8n555+foVOu4i59reFBH3zzzTd2CFN++OGHDI9lVwfg7HDHHXfYE4bK8OuDHWrevLl5+umnjQ905dC+fXvju8WLF5vTTz890+ZQvuQfaDRQr8ecOXPsLfFz4UvwkQrHKVWxXrRokSlbtqwNpp599lm7/Y8//rBTS75QO5D4EY+QOq5HoTpoVAOKg5EWwcdnn31mUkGqPI+FCxea119/fZ/tRx11lNm4caPxwciRI00q0BWerlQrVaqUYbvyPXSl5wPfk01T6fOt/l0XX3yxDT4UMClvQmbMmGGqV69ufNG0aVM7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1662nHTnzp3t/1EQ9dJLL5ly5cqZhQsX2se1TX8PAI4p5wNA6ho8eHBQsWJFe3/btm1B7ty5g1GjRsUe37VrV1CuXLngkUcesV/37ds3OPbYY+32zDRr1izo2bNnhm0XXHBB0LlzZ3t/7NixQZEiRYItW7b8433/5ZdflJMWLFy40H69fPly+/UTTzyR4fsqV64cvP766xm23X///UGjRo2y/NmPPvpoULdu3X+8jwAOHiMfQBrRFIamGho3bhzbpnyJBg0amO+++85+rVEGTbMcah7F2WefbUdUNLXRqlUre9PohaY7/s6SJUvsCMaMGTPMxo0bzd69e+12TRvVqlUr9n316tXLMFqi59WlSxc7mhHSSE18jwtN1QwZMsR+r0ZV9LhGdQC4R84HgAzy58+/38c1/ZG4SC4+d0JTGXPnzjVvvPGGTXRVMHHSSSeZTZs2/e3vbtu2rZ2meeGFF2wAopvs2rUrw/fFd+dUICH6Pwqcwts333xjpk+fbh+bNm2a6dSpkzn33HPNBx98YKdl7rrrrn1+LgA3CD6ANKL8jzx58tgcjvjAQfkgNWvWtF8r30I5GlklYx555JFm7dq1sa//+usve6KPp6W8LVq0sPkkCxYssImiyh/Zn19//dUsXrzY9OvXz5x11lk2efT333//2+ekhFblcfz444+mSpUqGW7HHnus/Z6pU6fa0RgFHBo1UTLsypUr//ZnA8geTLsAaUQjBjfeeKO59dZbbXJphQoVbIDwxx9/2GkL6d69u3nqqadMx44dTd++fe3UhUYQNDVTrVo107x5c5sUOmHCBBvMPP744xlGNTSyoEBASavFixc3//3vf+30if7v/uh7tcJFybEaMdFUixJLD8S9995rk1y1r5rm0YoeJb4qeNG+KtjQzxs9erSpX7++3fdx48b9w78mgEN2CHkiADxNOJU///wz6NGjR1CqVKkgb968QePGjYOZM2dm+D/z588PzjnnnKBAgQJB4cKFg6ZNmwbLli2zjykR9cYbbwxKlCgRHHXUUcHAgQMzJJx+8cUXNim1ePHiQf78+YPatWsHY8aMOaB9/eSTT4IaNWrY/dL/mzx5sk0wHTduXIaE06+//nqf/6sk2jp16gR58uSxv/v0008P3nnnndjjt956a1CyZMmgUKFCwSWXXGL/LkWLFj3EvyqAf4IKpwAAwClyPgAAgFMEHwCcUM5FoUKFsrzpcQDpgWkXAE6oroZWvWRFpd9peAekB4IPAADgFNMuAADAKYIPAADgFMEHAABwiuADAAA4RfABAACcIvgAAABOEXwAAACnCD4AAIBx6f8BL/9W1RgzcOYAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "top_focus = df['focus_area'].value_counts().nlargest(10)\n",
    "top_focus.plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "ead54376-6929-4ceb-a083-96b12e11b40d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty DataFrame\n",
      "Columns: [question, answer, source, focus_area, question_len, answer_len]\n",
      "Index: []\n"
     ]
    }
   ],
   "source": [
    "print(df[df['answer'].isnull()])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "c321577a-d294-4ecd-aaeb-bdd7d9d82b07",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Short questions:\n",
      " Empty DataFrame\n",
      "Columns: [question, answer]\n",
      "Index: []\n",
      "Short answers:\n",
      " Empty DataFrame\n",
      "Columns: [question, answer]\n",
      "Index: []\n"
     ]
    }
   ],
   "source": [
    "# Questions or answers with very few words (e.g., less than 3)\n",
    "short_questions = df[df['question'].fillna('').apply(lambda x: len(x.split()) < 3)]\n",
    "short_answers = df[df['answer'].fillna('').apply(lambda x: len(x.split()) < 3)]\n",
    "\n",
    "print(\"Short questions:\\n\", short_questions[['question', 'answer']])\n",
    "print(\"Short answers:\\n\", short_answers[['question', 'answer']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "f00cb244-1a97-47eb-9258-730a33ea5f80",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df[df['answer'].fillna('').apply(lambda x: len(x.split()) >= 3)]\n",
    "df = df[df['question'].fillna('').apply(lambda x: len(x.split()) >= 3)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "2c9e8ff6-d924-4d38-95f2-354bf7a9cf08",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Missing questions: 0\n",
      "Missing answers: 0\n"
     ]
    }
   ],
   "source": [
    "missing_questions = df[df['question'].isnull()]\n",
    "missing_answers = df[df['answer'].isnull()]\n",
    "\n",
    "print(f\"Missing questions: {len(missing_questions)}\")\n",
    "print(f\"Missing answers: {len(missing_answers)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "75a23df8-d040-4d1f-8720-3c0ad017f7f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# duplicates = df[df.duplicated(subset=['question', 'answer'], keep=False)]\n",
    "# print(f\"Number of duplicate question-answer pairs: {len(duplicates)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "494fda75-781d-4c51-82ae-325dc91426be",
   "metadata": {},
   "outputs": [],
   "source": [
    "# print(duplicates.index.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "431de33a-01a1-4225-9a97-462a383df226",
   "metadata": {},
   "outputs": [],
   "source": [
    "# multi_answers = df.groupby('question').filter(lambda x: x['answer'].nunique() > 1)\n",
    "# print(multi_answers[['question', 'answer']].sort_values(by='question'))\n",
    "# #"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "7ae97363-23ad-4a1f-bac8-8864ecb5cd0c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Potentially noisy answers: 0\n",
      "Empty DataFrame\n",
      "Columns: [question, answer]\n",
      "Index: []\n"
     ]
    }
   ],
   "source": [
    "import re\n",
    "\n",
    "# Answers with less than 50% alphabetic characters (maybe gibberish or code)\n",
    "def alpha_ratio(text):\n",
    "    text = str(text)\n",
    "    if len(text) == 0:\n",
    "        return 0\n",
    "    alphabets = len(re.findall(r'[a-zA-Z]', text))\n",
    "    return alphabets / len(text)\n",
    "\n",
    "noisy_answers = df[df['answer'].apply(alpha_ratio) < 0.5]\n",
    "print(f\"Potentially noisy answers: {len(noisy_answers)}\")\n",
    "print(noisy_answers[['question', 'answer']].head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "19bf96f9-479b-4c6e-a271-72908226fa64",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df[df['answer'].apply(alpha_ratio) >= 0.5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "bc86106a-e53c-49b9-a9a8-c7af8a624109",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjAAAAGzCAYAAAAxPS2EAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8ekN5oAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA08klEQVR4nO3dDZzNdf7//9cMM8a4GNfXV1OIkBa7SIq4kbAJu0XJhlolJWIbt1Zb35ZStrSu6ttt09aWXLeusy5iIytFqLA1ohgzLmYwmHHx+d9e79//c77njMEMh/N5f87jfrudPvP5fN5zzuecoznP876McRzHEQAAAIvERvoCAAAACosAAwAArEOAAQAA1iHAAAAA6xBgAACAdQgwAADAOgQYAABgHQIMAACwDgEGAABYhwADwFq7d++WmJgYefXVV6/ZY06fPt08pj721fa73/1O6tSpE7Hn+6c//ck8HuBFBBhARKZMmWL+ULds2VKiUbt27aRx48biVYsXLzYfpuG2evVq8767t2LFiknlypXN6zF27FjJyMgIy+OcOHHCXL8+ntd4+dqAiyHAACLyj3/8w3zT/c9//iP//e9/I305yCfAPP/881ft/p944gl577335K233pKRI0dKuXLl5LnnnpOGDRvKypUrQ8r269dPTp48KbVr1y5USNDrL2xI+N///V/ZsWOHXE0Xu7Znn33WPFfAi4pG+gKASEtNTZV169bJ3Llz5fe//70JM/rh5Sfnzp2T3NxcSUhIiPSleFLbtm2ld+/eIce2bNkinTp1kl69esk333wjVatWNceLFClibldTdna2lChRQuLi4iSSihYtam6AF1EDg6ingaVs2bLStWtX8yGm+3kF9z3Qb+nXX3+9aW745S9/KRs3bgwpm5aWJg899JDUqFHDlNEPvrvvvjvQZ2L48OFSvnx5CV4IfujQoeb+33jjjcCxAwcOmGNTp04NHMvJyTHhqm7duua+a9asKaNGjTLHg+nvPf744+a5NGrUyJRdunTpFb9WS5YsMR/2+uFaqlQp85pt3779vH4bJUuWlJ9//ll69Ohhfq5YsaI8/fTTcvbs2ZCyhw4dMjUapUuXljJlykj//v1NcNDr174m7v1Nnjw58LzcW16Xel8Kq2nTpvL6669LZmamTJo06aJ9YL744gvp3LmzVKhQQYoXLy7JyckyYMAA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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "df['answer_len'] = df['answer'].fillna('').apply(lambda x: len(x.split()))\n",
    "\n",
    "plt.boxplot(df['answer_len'])\n",
    "plt.title('Answer Length Distribution')\n",
    "plt.show()\n",
    "\n",
    "# # Consider filtering out answers shorter than 3 or longer than, say, 500 words\n",
    "# filtered_df = df[(df['answer_len'] >= 3) & (df['answer_len'] <= 500)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "6af4d723-4d50-44d8-9aff-8cbb8f5f98f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# # 5. Display the first 3 questions and answers for each focus_area\n",
    "# for area in df_med['focus_area'].unique():\n",
    "#     print(f\"Focus Area: {area}\")\n",
    "#     display(df_med[df_med['focus_area'] == area][['question', 'answer']].head(3))\n",
    "#     print(\"\\n\" + \"=\"*50 + \"\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "67e8efbb-e62e-413c-ba55-3332ae02fd49",
   "metadata": {},
   "outputs": [],
   "source": [
    "# # 6. Find and display the 20 most common words in the 'question' column (after removing stopwords and punctuation)\n",
    "\n",
    "# from collections import Counter\n",
    "# import nltk\n",
    "# from nltk.corpus import stopwords\n",
    "# import string\n",
    "\n",
    "# # Download stopwords if not already downloaded\n",
    "# nltk.download('stopwords')\n",
    "\n",
    "# stop_words = set(stopwords.words('english'))\n",
    "\n",
    "# # Combine all questions into one large string, convert to lowercase\n",
    "# all_questions = \" \".join(df_med['question'].fillna('').str.lower())\n",
    "\n",
    "# # Remove punctuation\n",
    "# all_questions = all_questions.translate(str.maketrans('', '', string.punctuation))\n",
    "\n",
    "# # Split into words\n",
    "# words = all_questions.split()\n",
    "\n",
    "# # Remove stopwords\n",
    "# filtered_words = [word for word in words if word not in stop_words]\n",
    "\n",
    "# # Count the most common words\n",
    "# word_counts = Counter(filtered_words)\n",
    "# most_common_words = word_counts.most_common(20)\n",
    "\n",
    "# print(\"Top 20 most common words in questions:\")\n",
    "# for word, count in most_common_words:\n",
    "#     print(f\"{word}: {count}\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
