#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 install pyspark')
get_ipython().system('pip3 install graphframes')


# In[ ]:


import pyspark as ps
import os
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages graphframes:graphframes:0.7.0-spark2.4-s_2.11 pyspark-shell"
)


# ### Usefull functions

# In[ ]:


def init_spark(app_name, master_config):
    """
    :params app_name: Name of the app
    :params master_config: eg. local[4]
    :returns SparkContext, SQLContext, SparkSession:
    """
    conf = (ps.SparkConf().setAppName(app_name).setMaster(master_config))

    sc = ps.SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    sql_ctx = SQLContext(sc)
    spark = SparkSession(sc)

    return (sc, sql_ctx, spark)


# In[ ]:


@udf(returnType=BooleanType())
def check_map(dist_map):
    return bool(dist_map)


# In[ ]:


def get_id_by_title(title, vertices):
    return vertices.filter(vertices['title']==title).head()[0]


# ### Queries

# In[ ]:


def first_query(g, year, art_id):
    sp = g.shortestPaths(landmarks=[art_id])
    results = sp.filter(sp['year']==year)
    results = results.withColumn("isLinked", check_map("distances"))
    results = results.filter(results['isLinked'])
    results = results.drop(*['distances','isLinked'])
    return results


# In[ ]:


def second_query(g, art_id):
    sp = g.shortestPaths(landmarks=[art_id])
    sp = sp.withColumn("isLinked", check_map("distances"))
    popular_by_year = sp.filter(sp['isLinked']).groupBy("year").count()
    return popular_by_year.orderBy(popular_by_year['count'].desc())


# In[ ]:


def third_query(g, rp=0.15, mi=10):
    query_result = g.pageRank(resetProbability=rp, maxIter=mi).vertices
    return query_result.orderBy(query_result['pagerank'].desc())


# In[ ]:


def lab_prop(g, mi=10):
    communities = g.labelPropagation(maxIter=10)
    com = communities.groupBy('label').count()
    return com.orderBy(com['count'].desc()), communities


# ### Create spark env

# In[ ]:


sc, sql_ctx, spark = init_spark("App_name", "local[*]")


# ## Work with small dataset
# ### Reading

# In[ ]:


path_to_vertices = "/home/kostin_001/citation-data/small/v.txt"
path_to_edges = "/home/kostin_001/citation-data/small/e.txt"


# In[ ]:


vertexScheme = StructType([
  StructField("id", StringType(), False),
  StructField("title", StringType(), False),
  StructField("year", IntegerType(), False),
  StructField("publication_venue", StringType(), False),
  StructField("authors", StringType(), False)]
)


# In[ ]:


edgeScheme = StructType([
  StructField("src", StringType(), False),
  StructField("dst", StringType(), False),
  StructField("count", IntegerType(), False)]
)


# In[ ]:


vertices = spark.read.option("delimiter", "\t").option("schema", vertexScheme).option("header", False).csv(path_to_vertices).toDF("id", "title","year", "publication_venue", "authors")


# In[ ]:


edges = spark.read.option("delimiter", "\t").option("schema", edgeScheme).option("header", False).csv(path_to_edges).toDF("src", "dst","count")
edges = edges.drop('count')


# ## Creating GraphFrame

# In[ ]:


from graphframes import *
g = GraphFrame(vertices,edges)


# ## Queries

# Return all the papers that were written in 2001 and can be traced back (through citations, direct or indirect) to the paper ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging.

# In[ ]:


title = 'ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging'
art_id = get_id_by_title(title, vertices)
year = '2001'


# In[ ]:


first_query(g, year, art_id).show()


# On which year, there is the most papers that trace back to the paper mentioned above?

# In[ ]:


second_query(g, art_id).show()


# Return the most influential papers in the citation graph.

# In[ ]:


third_query(g).show(1)


# ## Label prop

# In[ ]:


com, communities = lab_prop(g)
com.show(5)


# In[ ]:


top5 = com.orderBy(com["count"].desc()).collect()[:5]
for i in range(5):
    communities[communities['label']==top5[i]["label"]].show()


# ## Working with large dataset

# In[ ]:


path_to_vertices_title = "/home/kostin_001/citation-data/large/paper_title.tsv"
path_to_vertices_year = "/home/kostin_001/citation-data/large/paper_year.tsv"
path_to_edges = "/home/kostin_001/citation-data/large/ref.tsv"


# In[ ]:


vertices_large = spark.read.option("delimiter", "\t").option("header", True).csv(path_to_vertices_title).toDF("id", "title")
vertices_l_title = spark.read.option("delimiter", "\t").option("header", True).csv(path_to_vertices_year).toDF("id", "year")


# In[ ]:


edges_large = spark.read.option("delimiter", "\t").option("header", True).csv(path_to_edges).toDF("src", "dst")


# ### Preprocess vertices
# #### Join and remove empty vertices (with empty title or empty year)

# In[ ]:


vertices_large = vertices_large.filter(vertices_large['title']!='\\N')


# In[ ]:


vertices_large = vertices_large.join(vertices_l_title, "id", how='inner')


# ## Creating GraphFrame

# In[ ]:


g_large = GraphFrame(vertices_large,edges_large)


# ## Queries

# Return all the papers that were written in 2001 and can be traced back (through citations, direct or indirect) to the paper Machine Learning.

# In[ ]:


title='Machine Learning'
art_id = get_id_by_title(title, vertices_large)
year = '2001'


# In[ ]:


first_query(g_large, year, art_id).show()


# On which year, there is the most papers that trace back to the paper mentioned above?

# In[ ]:


second_query(g_large, art_id).show()


# Return the most influential papers in the citation graph.

# In[ ]:


third_query(g_large).show(1)


# ## Label prop

# In[ ]:


com, communities = lab_prop(g_large)
com.show(5)


# In[ ]:


top5 = com.orderBy(com["count"].desc()).collect()[:5]
for i in range(5):
    communities[communities['label']==top5[i]["label"]].show()


# In[ ]:


spark.stop()

