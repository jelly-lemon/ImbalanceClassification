本目录包含了用到的数据集，读取数据的相关函数

# 数据集命名规则
原始数据集名-负样本类别下标号 vs. 正样本类别下标号   
如：iris-0 表示类别下标为0的类作为负样本，其余全部作为正样本

# 数据集来源 
KEEL：https://sci2s.ugr.es/keel/category.php?cat=clas  
UCI：http://archive.ics.uci.edu/ml/datasets.php  


# 数据集信息



## 1 < IR <= 5
|data sets              |class  |attributes     |IR             |Other
|----                   |----   |----           |----           |----  
|banana-1	            |2	    |2	            |2924/2376=1.23 | 
|australian-1	        |2	    |14	            |112/90=1.25    | 
|bupa-0	                |2	    |6	            |124/90=1.38    | 
|bands-0	            |2	    |19	            |153/90=1.70    | 
|iris-0                 |3      |4              |100/50=2.00    |数据太简单了，分类正确率都快到100%了 
|glass-0	            |7	    |9	            |144/70=2.06    | 
|tae-0	                |3	    |5	            |102/49=2.08    | 
|titanic-1	            |2	    |3	            |188/90=2.10    | 
|yeast-1                |10     |8              |1055/429=2.46  |
|ecoli-1	            |8	    |7	            |259/77=3.36    |
|spectfheart-0	        |2	    |44	            |212/55=3.85    |属性过多  
|appendicitis-1	        |2	    |7	            |85/21=4.05     |负样本数量较少
|zoo-1	                |7	    |16	            |81/20=4.05     |数据太多简单
|yeast-0-6	            |10	    |8	            |1205/279=4.32  |
|cleveland-1	        |5	    |13	            |243/54=4.50    |




## 5 < IR <= 10
|data sets              |class  |attributes     |IR             |Other
|----                   |----   |----           |----           |----  
|yeast-0                |10     |8              |457/90=5.08    |  
|hepatitis-0	        |2	    |19	            |67/13=5.15     |负样本数量太少
|ecoli-7	            |8	    |7	            |284/52=5.46    |
|newthyroid-0	        |3	    |5	            |185/30=6.17    | 
|zoo-3	                |7	    |16	            |88/13=6.77     |数据太过简单
|winequality-red-7	    |11	    |11	            |1400/199=7.04  |没有0，1，2这几个类
|fertility-1	        |2	    |9	            |88/12=7.33     |负样本量有点少
|cleveland-2	        |5	    |13	            |262/35=7.49    | 
|ecoli-4	            |8	    |7	            |301/35=8.60    | 
|page-blocks-1-2-3-4	|5	    |10	            |791/90=8.79    |
|zoo-6	                |7	    |16	            |91/10=9.10     |负样本太少
|vowel-0	            |11	    |13	            |900/90=10.00   |数据简单，分类容易

## 10 < IR <= 15
|data sets              |class  |attributes |IR             |Other
|----                   |----   |----       |----           |----  
|ecoli-2-3-5-6	        |8	    |7	        |307/29=10.59   |                   
|page-blocks-1-2-3	    |5	    |10	        |1019/90=11.32  | 
|glass-2	            |7	    |9	        |197/17=11.59   |负样本数量较少
|zoo-5	                |7	    |16	        |93/8=11.62     |负样本太少
|balance-1	            |3	    |4	        |576/49=11.76   | 
|cleveland-4 vs. 0	    |2	    |13	        |160/13=12.31   | 
|movement_libras-1	    |15	    |90	        |336/24=14.00   |属性数量有点多 
|yeast-7 vs. 1	        |2	    |8	        |429/30=14.30   | 

## 15 < IR <= 20
|data sets              |class  |attributes |IR             |Other
|----                   |----   |----       |----           |----  
|ecoli-5	            |8	    |7	        |316/20=15.80   | 
|zoo-2	                |7	    |16	        |96/5=19.20     |样本数量太少了

## 20 < IR <= 25
|data sets              |class  |attributes |IR             |Other
|----                   |----   |----       |----           |----  
|cleveland-4	        |5	    |13	        |284/13=21.85   |负样本太少
|yeast-7 vs. 1-4-5-8    |5	    |8	        |663/30=22.10   |
|zoo-4	                |7	    |16	        |97/4=24.25     |负样本太少
|letter-img-0	        |26	    |16	        |2191/90=24.35  |没有0，1，2这几个类
|isolet5-7	            |26	    |617	    |1499/60=24.98  |每个类数量都是60
|isolet1+2+3+4-0	    |26	    |617	    |2249/90=24.99  |每个类都是90个


## 25 < IR <= 30
|data sets              |class  |attributes |IR             |Other
|----                   |----   |----       |----           |----  
|letter-img-1	        |26	    |16	        |2259/90=25.11  | 
|yeast-4	            |10	    |8	        |1433/51=28.10  | 
|winequality-red-4	    |11	    |11	        |1546/53=29.17  | 

## 30 < IR <= 35



## 35 < IR <= 40
|data sets                  |class  |attributes |IR             |Other
|----                       |----   |----       |----           |----  
|winequality-red-8 vs. 6	|2	    |11	        |638/18=35.44   |

## 40 < IR <= 50
|data sets                  |class  |attributes |IR             |Other
|----                       |----   |----       |----           |----  
|yeast-6	                |10	    |8	        |1449/35=41.40  | 

## 60 < IR <= 65

|data sets              |class  |attributes |IR             |Other
|----                   |----   |----       |----           |----  
|page-blocks-3	        |5	    |10	        |5385/87=61.90  |

## 65 < IR <= 70
|data sets              |class  |attributes |IR             |Other
|----                   |----   |----       |----           |----  


## 80 < IR <= 100
|data sets              |class  |attributes |IR             |Other
|----                   |----   |----       |----           |----  
|winequality-red-8	    |11	    |11	        |1581/18=87.83  |没有0，1，2这几个类

## IR > 100
|data sets              |class  |attributes |IR             |Other
|----                   |----   |----       |----           |----  
|winequality-red-3	    |11	    |11	        |1589/10=158.90 |没有0，1，2这几个类

## 黑名单
|data sets              |class  |attributes |IR             |Other
|----                   |----   |----       |----           |----  
|coil2000-1	            |2	    |85	        |1418/90=15.76  |属性数量有点多 
			