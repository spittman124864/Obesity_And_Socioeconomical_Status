{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0",
    "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy.random as np\n",
    "import sys\n",
    "import matplotlib \n",
    "import seaborn as sns\n",
    "import numpy as np\n",
    "from subprocess import check_output\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "%matplotlib inline\n",
    "\n",
    "\n",
    "import os\n",
    "print(os.listdir(\"../input\"))\n",
    "\n",
    "# Any results you write to the current directory are saved as output."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "_cell_guid": "5db2907a-2700-4cae-9ba7-e142a762d2e5",
    "_uuid": "5e7945f70becff8b365b1a3830f648b0ad94c332",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Location of file\n",
    "Location = '../input/Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv'\n",
    "\n",
    "df = pd.read_csv(Location)\n",
    "\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "03c501c3-1e18-4386-89da-96d4932863d5",
    "_uuid": "cb43fad92fac926a737669aaee61804a873c2c36",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "_cell_guid": "f7716010-c81a-4861-a285-6cba0f8642fb",
    "_uuid": "d210dae4a863b5e4c440af1f51d5c77a64278731",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Getting Rid of All Extraneous Info\n",
    "\n",
    "df.drop(['Low_Confidence_Limit','High_Confidence_Limit ','YearEnd','Topic','Class','Datasource','Data_Value_Unit','QuestionID','ClassID','TopicID','DataValueTypeID','Data_Value_Type','Data_Value_Footnote_Symbol','Data_Value_Footnote','StratificationCategoryId1','StratificationID1'],1);\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "_cell_guid": "3f228663-20eb-47e8-9885-d3a3ee03046e",
    "_uuid": "d60952e765d06955af947fedb7966cb63d32c0f1",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Create separate Dataform from df by gender df2, by education level dfedu, and by income dfedu\n",
    "\n",
    "df2=df[(df['Stratification1']=='Male')|(df['Stratification1']=='Female')]\n",
    "dfedu=df[df['StratificationCategory1']=='Education']\n",
    "dfinc=df[df['StratificationCategory1']=='Income']\n",
    "\n",
    "#reset index for each of the new dataforms\n",
    "\n",
    "df2 = df2.reset_index(drop = True)\n",
    "dfedu = dfedu.reset_index(drop = True)\n",
    "dfinc = dfinc.reset_index(drop = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "_cell_guid": "d9580d4c-9bd7-43fd-adeb-a17ecebf34af",
    "_uuid": "7247779c99e552bd00919276341b46273670b52a",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Each category has the same survey questions\n",
    "\n",
    "df2['Question'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "_cell_guid": "f382e766-6bae-4ffb-93e1-eb66eb0dabf4",
    "_uuid": "11f0f94d522ac373242dedfbed64abb70eb0ff61",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#here we are interested in the survey question directly about obesity and overweight percent\n",
    "\n",
    "X=['Percent of adults aged 18 years and older who have obesity','Percent of adults aged 18 years and older who have an overweight classification']\n",
    "\n",
    "\n",
    "df2=df2[df2['Question']==X[0]]\n",
    "\n",
    "#In case we wanted both. df3=df2[df2['Question'].apply(lambda x: x in X)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "_cell_guid": "fe96a301-113c-4bff-98f8-8c3aebc80f91",
    "_uuid": "897bb7690718436b255582050a9f5bcc636ab40c",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#survey data covers 2011 - 2014 (all states) or 2016 most states\n",
    "#choose 2014 since it has the most data \n",
    "\n",
    "df2=df2[df2['YearStart']==2014]\n",
    "\n",
    "#separate out national so that we can calculate the national obesity rate for 2014\n",
    "df2n=df2[(df2['LocationDesc']=='National')]\n",
    "\n",
    "#Cut out terriotories that our not included within 50 states + DC data\n",
    "df2=df2[~(df2['LocationDesc']=='National')]\n",
    "df2=df2[~(df2['LocationDesc']=='Guam')]\n",
    "df2=df2[~(df2['LocationDesc']=='Puerto Rico')]\n",
    "df2['LocationDesc'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "_cell_guid": "1761f475-05b6-44c9-a5fd-829db89fdd36",
    "_uuid": "72162dab5b70d681a85420fa9b4a8d476e5e2234",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#group data by state and take the mean of men and women rates for each state\n",
    "\n",
    "sorted_df = df2.sort_values(['LocationDesc'], ascending = [True])\n",
    "sorted_df=sorted_df[['LocationAbbr','LocationDesc','Data_Value','Gender']]\n",
    "sorted_df = sorted_df.groupby('LocationDesc', as_index=False).mean()\n",
    "\n",
    "#calculate the average (over men and women) obesity rate for the country\n",
    "\n",
    "natmeanobesity2014=sum(df2n['Data_Value'])/len(df2n)\n",
    "print(natmeanobesity2014)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "_cell_guid": "04c095a8-5e88-4e84-aea4-4693bdb7ac59",
    "_uuid": "cf1af7f72ebe309ec29e8a93a1c93d393b36113a",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Let's plot a bar graph of the most and least obese states in the US\n",
    "#Mark the national average in red\n",
    "\n",
    "#For those that have LaTex\n",
    "#plt.rc('text', usetex=True)\n",
    "\n",
    "plt.rc('font', family='serif')\n",
    "plt.rcParams.update({'font.size': 14})\n",
    "\n",
    "sorted_df = sorted_df.sort_values(['Data_Value'], ascending = [True])\n",
    "\n",
    "plt.figure(figsize = (10,16))\n",
    "\n",
    "plt.subplot(2,1,1)\n",
    "ax=sns.barplot(y=sorted_df.tail(10).LocationDesc,x=sorted_df.tail(10).Data_Value,palette=\"Blues_d\")\n",
    "ax.set_ylabel('US State')\n",
    "ax.set_xlabel('Obesity Rate (%)')\n",
    "ax.set_title('10 Most Obese States in 2014')\n",
    "\n",
    "plt.plot([natmeanobesity2014,natmeanobesity2014],[-1,10], '--',color = 'r')\n",
    "\n",
    "plt.subplot(2,1,2)\n",
    "ax=sns.barplot(y=sorted_df.head(10).LocationDesc,x=sorted_df.head(10).Data_Value,palette=\"Blues_d\")\n",
    "ax.set_ylabel('US State')\n",
    "ax.set_xlabel('Obesity Rate (%)')\n",
    "ax.set_title('10 Least Obese States in 2014')\n",
    "\n",
    "plt.plot([natmeanobesity2014,natmeanobesity2014],[-1,10], '--',color = 'r')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "5d0a5519-8231-4aa2-a5f2-0b9c066d18e1",
    "_uuid": "adc0bba133cdaa872888cdafcb5e1af7ee919ef0",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "c0e0a630-0ba2-4c9e-a8ce-06675d8e2b14",
    "_kg_hide-output": true,
    "_uuid": "e5184812f402890774f974e9a35e6d59d1e31e40",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "!pip install plotly\n",
    "\n",
    "sorted_df = df2.sort_values(['LocationDesc'], ascending = [True])\n",
    "sorted_df=sorted_df[['LocationAbbr','LocationDesc','Data_Value','Gender']]\n",
    "sorted_df2 = sorted_df.groupby('LocationAbbr', as_index=False).mean()\n",
    "#Let's make a map to see the geographic locations of the obesity rates\n",
    "\n",
    "import plotly.plotly as py\n",
    "\n",
    "\n",
    "scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\\\n",
    "            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]\n",
    "\n",
    "\n",
    "data = [ dict(\n",
    "        type='choropleth',\n",
    "        colorscale = 'YlOrRd',\n",
    "        autocolorscale = False,\n",
    "        reversescale = True,\n",
    "        locations = sorted_df2['LocationAbbr'],\n",
    "        z = sorted_df2['Data_Value'].astype(float),\n",
    "        locationmode = 'USA-states',\n",
    "        text = sorted_df2['LocationAbbr'],\n",
    "        marker = dict(\n",
    "            line = dict (\n",
    "                color = 'rgb(255,255,255)',\n",
    "                width = 2\n",
    "            ) ),\n",
    "        colorbar = dict(\n",
    "            title = \"% Obesity\")\n",
    "        ) ]\n",
    "\n",
    "layout = dict(\n",
    "        #title = '2011 US Agriculture Exports by State<br>(Hover for breakdown)',\n",
    "        geo = dict(\n",
    "            scope='usa',\n",
    "            projection=dict( type='albers usa' ),\n",
    "            showlakes = True,\n",
    "            lakecolor = 'rgb(255, 255, 255)'),\n",
    "             )\n",
    "    \n",
    "fig = dict( data=data, layout=layout )\n",
    "py.iplot( fig, filename ='somename' )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "3f4b8804-e933-4351-bdd6-88bf568b5586",
    "_uuid": "bece6552b75c56345c434b1a366c258bb37f816d",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Now let's explore if obesity is somehow correlated with education level\n",
    "\n",
    "dfedu.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "61d6796c-9ac6-4eee-8667-97167d7d55c6",
    "_uuid": "ba76108bfaa60005796ab1909163ba815b8390fa",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Just like for the gender one, we need to isolate just the obesity question and the year\n",
    "\n",
    "\n",
    "X=['Percent of adults aged 18 years and older who have obesity','Percent of adults aged 18 years and older who have an overweight classification']\n",
    "\n",
    "#df3=df2[df2['Question'].apply(lambda x: x in X)]\n",
    "dfedu=dfedu[dfedu['Question']==X[0]]\n",
    "dfedu=dfedu[dfedu['YearStart']==2014]\n",
    "\n",
    "#Cut out all locations that aren't within the 50 states + DC\n",
    "dfedu=dfedu[~(dfedu['LocationDesc']=='National')]\n",
    "dfedu=dfedu[~(dfedu['LocationDesc']=='Guam')]\n",
    "dfedu=dfedu[~(dfedu['LocationDesc']=='Puerto Rico')]\n",
    "dfedu['LocationDesc'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "ffecae36-cdf1-40de-a3fe-4a54f426839c",
    "_uuid": "f1badacafaa9755d6d7fe58370a29406833ffe5e",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#select the four relevant columns to analyze the obesity rate versus educational level\n",
    "\n",
    "dfedu = dfedu.reset_index(drop = True)\n",
    "dfedu=dfedu[['YearStart','LocationDesc','Data_Value','Education']]\n",
    "dfedu.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "2269d4e2-e1e8-4b08-be1d-d8f56225238e",
    "_uuid": "ec5449cd0aad8fa400960c33be143b42929d785c",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Create a list of all 4 educational levels\n",
    "ledu=dfedu.Education.unique()\n",
    "\n",
    "\n",
    "#want to treat education levels as dummy variables, so this assigns 1 or 0 depending on the group\n",
    "for i in ledu:\n",
    "    dfedu[i]=dfedu['Education'].apply(lambda x: int(x==i))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "fbfd5485-67ef-4dc7-b75c-c7c8388fcd96",
    "_uuid": "274130197e911649af378a3e7e4728246760918b",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#select the four relevant columns to analyze the obesity rate versus educational level\n",
    "\n",
    "dfedu = dfedu.reset_index(drop = True)\n",
    "dfedu=dfedu[['YearStart','LocationDesc','Data_Value','Education']]\n",
    "dfedu.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "d03acb32-0bc7-4e77-960b-c2b01739f0c4",
    "_uuid": "b2ca2e761fcfef0d38162875e69f84ec6316d009",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Create a list of all 4 educational levels\n",
    "ledu=dfedu.Education.unique()\n",
    "\n",
    "\n",
    "#want to treat education levels as dummy variables, so this assigns 1 or 0 depending on the group\n",
    "for i in ledu:\n",
    "    dfedu[i]=dfedu['Education'].apply(lambda x: int(x==i))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "a7da5407-1faa-472e-9c30-1c0f323ba376",
    "_uuid": "ac601fd8cb9b8a41ba9802ce2132446ddf50efb7",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Let's just take a look at how the obesity varies amongst states for those without highschool\n",
    "#From the bar graph below, it is clear that Wyomming is an outlier. \n",
    "\n",
    "#plt.rc('text', usetex=True)\n",
    "plt.rcParams.update(plt.rcParamsDefault)\n",
    "plt.rc('font', family='serif')\n",
    "plt.rcParams.update({'font.size': 14})\n",
    "\n",
    "\n",
    "\n",
    "dfeduLHS=dfedu[dfedu[ledu[0]]==1]\n",
    "dfeduLHS = dfeduLHS.reset_index(drop = True)\n",
    "\n",
    "plt.figure(figsize = (10,16))\n",
    "ax=sns.barplot(y=dfeduLHS.LocationDesc,x=dfeduLHS.Data_Value,palette=\"Blues_d\")\n",
    "ax.set_ylabel('US State')\n",
    "ax.set_xlabel('Obesity Rate for People Without Highschool education (%)')\n",
    "\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "edacd641-f784-4295-bfa3-c69b62cd5b1d",
    "_uuid": "3f25343a9d764afdf14e57f7803679247ad85f9d",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Let's see how obesity compares for those with a HS education\n",
    "\n",
    "dfeduHS=dfedu[dfedu[ledu[1]]==1]\n",
    "dfeduHS = dfeduHS.reset_index(drop = True)\n",
    "\n",
    "\n",
    "plt.figure(figsize = (10,16))\n",
    "ax=sns.barplot(y=dfeduHS.LocationDesc,x=dfeduHS.Data_Value,palette=\"Blues_d\")\n",
    "ax.set_ylabel('US State')\n",
    "ax.set_xlabel('Obesity Rate for People With Highschool Education (%)')\n",
    "\n",
    "\n",
    "#plt.plot(df.non_weighted_all_weekly, df.M_weekly,'o')\n",
    "#plt.plot(df.non_weighted_all_weekly, df.F_weekly,'o')\n",
    "#plt.legend(['Males','Females'])\n",
    "#plt.xlabel('Field Median Salary')\n",
    "#plt.ylabel('Salary')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "972151a2-bf3c-4992-9dd1-4df550b0185d",
    "_uuid": "5ae398ca1302f01af581b5917ab28311e50c0c9d",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Wyoming is an outlier AND has a small population and is very rural (i.e. different than other states) \n",
    "#I am going to take Wyoming it out of the Linear Regression Data Set without much harm\n",
    "\n",
    "#Note that since I have 4 dummy variables, need only three coefficients \n",
    "dfedu=dfedu[~(dfedu['LocationDesc']=='Wyoming')]\n",
    "model = LinearRegression()\n",
    "columns = dfedu.columns[5:8]\n",
    "X = dfedu[columns]\n",
    "\n",
    "X_std = StandardScaler().fit_transform(X)\n",
    "y = dfedu['Data_Value']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.18, random_state=42)\n",
    "\n",
    "model.fit(X_train,y_train)\n",
    "\n",
    "plt.barh([0,1,2],model.coef_)\n",
    "plt.yticks(range(3),dfedu.columns[4:7], fontsize = 10)\n",
    "plt.title('Regression Coefficients')\n",
    "\n",
    "plt.show()\n",
    "\n",
    "#Regression R^2 value shows that lack of education has a \"moderate\" effect on obesity rate\n",
    "print('R^2 on training...',model.score(X_train,y_train))\n",
    "print('R^2 on test...',model.score(X_test,y_test))\n",
    "\n",
    "print('Model Coefficients',model.coef_)\n",
    "print('Model Coefficients',model.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "_cell_guid": "1ba0a53c-e556-4ddb-8f57-500893d62b5c",
    "_uuid": "6288df81df21e5bf5c0e55ab88f55b9b59cf19fb",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "dfinc.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "_cell_guid": "3480ca04-377a-4743-b220-e523e20ae668",
    "_uuid": "077a2bf497c429d4d5c814b397acf7583cec154e",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Now let's look at the effect income has on obesity\n",
    "\n",
    "X=['Percent of adults aged 18 years and older who have obesity','Percent of adults aged 18 years and older who have an overweight classification']\n",
    "\n",
    "#df3=df2[df2['Question'].apply(lambda x: x in X)]\n",
    "dfinc=dfinc[dfinc['Question']==X[0]]\n",
    "dfinc=dfinc[dfinc['Question']==X[0]]\n",
    "dfinc=dfinc[dfinc['YearStart']==2014]\n",
    "dfinc=dfinc[~(dfinc['LocationDesc']=='National')]\n",
    "dfinc=dfinc[~(dfinc['LocationDesc']=='Guam')]\n",
    "dfinc=dfinc[~(dfinc['LocationDesc']=='Puerto Rico')]\n",
    "\n",
    "dfinc['LocationDesc'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "_cell_guid": "5300222b-567a-44e9-894c-798d997f1abd",
    "_uuid": "8b88c3886dd533deb62b1b212705de121ec0a03f",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "dfinc.Income.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "_cell_guid": "be501e1d-bc0a-4671-ba7f-9da01959cd6b",
    "_uuid": "fde5dfff0124c83d91a8c713ea16eb97e6150eaa",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "dfinc = dfinc.reset_index(drop = True)\n",
    "dfinc=dfinc[['YearStart','LocationDesc','Data_Value','Income']]\n",
    "\n",
    "linc=dfinc.Income.unique()\n",
    "\n",
    "#Create Dummy Variables from the income\n",
    "for i in linc:\n",
    "    dfinc[i]=dfinc['Income'].apply(lambda x: int(x==i))\n",
    "\n",
    "\n",
    "dfinc=dfinc[~(dfinc.Income=='Data not reported')]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "_cell_guid": "110d3991-5c15-4d00-9d30-6c81485b4598",
    "_uuid": "f02023dff667a20a237a5805e0b78c11d8ba1a81",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "dfinc.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "_cell_guid": "87d7b60e-18f8-44c2-8942-fa2c181bf01b",
    "_uuid": "b568cf94e9358addf62d2efa30c3da770ac1fd4b",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Let's test out whether income has an effect on obesity\n",
    "\n",
    "\n",
    "\n",
    "model = LinearRegression()\n",
    "columns = dfinc.columns[4:9]\n",
    "X = dfinc[columns]\n",
    "\n",
    "#X_std = StandardScaler().fit_transform(X)\n",
    "y = dfinc['Data_Value']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)\n",
    "\n",
    "model.fit(X_train,y_train)\n",
    "\n",
    "plt.figure(figsize = (10,10))\n",
    "plt.barh([0,1,2,3,4],model.coef_)\n",
    "plt.yticks(range(5),dfinc.columns[4:9], fontsize = 10)\n",
    "plt.title('Regression Coefficients')\n",
    "\n",
    "plt.show()\n",
    "\n",
    "#Yikes!  Such low regression coefficients illustrate that there is only a weak effect if any\n",
    "print('R^2 on training...',model.score(X_train,y_train))\n",
    "print('R^2 on test...',model.score(X_test,y_test))\n",
    "\n",
    "print('Model Coefficients',model.coef_)\n",
    "print('Model Coefficients',model.intercept_)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "_cell_guid": "19863556-711b-4251-8ec0-358d5e40c298",
    "_uuid": "6fb7e0de1f04b62fcba3bb3baf7f79a48aa37f25",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "dfavginc = dfinc.groupby('Income', as_index=False).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "_cell_guid": "270c0bb3-429b-4e09-a821-b504f69b33af",
    "_uuid": "1c95bbb2bc69157b8b1f89b7a579eb5f31f0d8a5",
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Let's look out the mean obesity rate for each income category to investigate\n",
    "\n",
    "dfavginc['IncomeOrder']=[1,2,3,4,5,0]\n",
    "\n",
    "\n",
    "sorted_df = dfavginc.sort_values(['IncomeOrder'], ascending = [True])\n",
    "plt.figure(figsize = (10,10))\n",
    "\n",
    "\n",
    "\n",
    "#National average marked in red\n",
    "#From this graph, we see that obesity doesn't monotonically decrease with increasing income.  \n",
    "##In a sense, these income categories really depend on the cost of living in each state.  \n",
    "#For example, $75K will buy you a lot in MS and places you in high income bracket, but not necessarily in MA   \n",
    "\n",
    "sorted_df.Data_Value.plot(kind='barh')\n",
    "\n",
    "plt.yticks(range(6),sorted_df.Income, fontsize = 12,family='serif')\n",
    "plt.plot([natmeanobesity2014,natmeanobesity2014],[-0.5,5.5], '--',color = 'r')\n",
    "plt.title('Mean Obesity By Income Group in 2014',family='serif')\n",
    "plt.xlim([0,38])\n"
   ]
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
   "version": "3.5.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
