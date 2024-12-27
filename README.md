# regression-api 
## Feb 2020 - Oct 2020, Role: Senior Research Programmer

Flask API is used to train regression models and results displayed on the dashboard.

### Code of conduct
Resources were used for specific literature/code, it is provided in the respective implementation file. The code provided here is implicitly expected not to be replicated for homeworks, assignments or any other programming work. It is welcome to take inspiration, but it is implicitly expected to cite this resource if used for learning, inspiration purposes. Please refer <a href="https://github.com/sushmaakoju/regression-api/blob/main/CODE_OF_CONDUCT.md">code of conduct</a>.

# regression-api-demo
This is a repository for Regression algorithms using scikit-learn and displays the results using Plotly/Dash interactive plots and a dashboard.

### Steps to install

Clone the repository:
```
git clone https://github.com/sushmaakoju/regression-api.git
cd regression-api
```

#### Start API (if pre-requisites are setup):
Assuming you already installed Python3.8.x, 
( Optional pre-requisites: Java and Spark/Hadoop setup) 
install following python requirements. 

- Make sure you have command prompt from cloned repository:

    ```
    cd regression-api or
    cd regression-api-master
    ```

- Install the requirements for the starting the flask api:

    ```
    pip install -r requirements.txt
    ```

- To start api on Windows, just execute following command from command prompt:

    ```
    start_api.bat
    ```

- To start api on Linux/MacOS, type following and 
    ```
    .\start_api.sh
    ```

- To test the api has successfully installed, open following url in browser:
    ```
    http://127.0.0.1:8050/
    ```
    You should see following displayed:
    ```
    {'result':"Welcome to Regression api!"}
    ```
- Once you start the api, you would have following urls for demo for test dataset:
    - http://127.0.0.1:8050/ 
    - http://127.0.0.1:8050/train_dtr
    - http://127.0.0.1:8050/train_svr
    - http://127.0.0.1:8050/train_lr
    - http://127.0.0.1:8050/train_rfr
    - http://127.0.0.1:8050/train_br

- To test the api using Postman, please follow the steps:
    - Download and install postman https://www.postman.com/downloads/
    - Configure requests in Postman as follows:
        - Create a New Collection "Regression-API" and create a New request with following url, request type
        <img src="https://user-images.githubusercontent.com/8979477/96185356-c63ce780-0f07-11eb-8a88-436925bc85ec.PNG" width="600" height="300">
        
        - Create request for each of following Api URLS:
            - http://127.0.0.1:8050/ 
            - http://127.0.0.1:8050/train_dtr
            - http://127.0.0.1:8050/train_svr
            - http://127.0.0.1:8050/train_lr
            - http://127.0.0.1:8050/train_rfr
            - http://127.0.0.1:8050/train_br 

        - Assuming you started api using start_api script from previous step, Click "Send" from postman for one of requests and see if request is successfully.
- Once you run any one of above results, navigate to http://127.0.0.1:8050/results/ to see Dashboard.
  The dashboard should look like this:
    <embed src="https://github.com/sushmaakoju/regression-api/blob/main/regression-api-dashboard.pdf" type="application/pdf" width="600" height="300">

    <img src="https://user-images.githubusercontent.com/8979477/96185358-c63ce780-0f07-11eb-84d0-d36cb289f772.PNG" width="600" height="300">

- Dataset credits:
    - https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/ 
    - https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength 


### Pre-requisites to Start the API
#### Install Spark 3.0.1

Pre-requisites:
- Java version 8+
- Python 3.8.x

##### Install Spark on Windows 10 Operating System

- Check Java version ()
    ``` 
    java -version
    ```

- If you don't have Java installed, download and install it from here:
    [Java 8x](https://java.com/en/download/manual.jsp)

- Download Spark 3.0.1 with Hadoop 2.7.4 version from [Spark](https://spark.apache.org/downloads.html)

- Verify checksum for Spark:

    ``` 
    certutil -hashfile complete-path-to-downloaded-spark-targz-file SHA512
    ```

- Compare checksum from [checksums section for spark 3.0.1 Hadoop 2.7.4](https://spark.apache.org/downloads.html). This should be same as displayed in certutil command.

- To install, create a Spark folder

    ``` cd \ 
        mkdir Spark 
        cd Spark 
    ```

- Extract downloaded-spark-targz-file named as "spark-3.0.1-bin-hadoop2.7.tar" to C:\Spark

- Download winutils.exe for Hadoop from [winutils for 2.7.4](https://github.com/cdarlint/winutils/tree/master/hadoop-2.7.4/bin)

- Create a folder Hadoop and copy the winutils.exe file in hadoop\bin folder. 
    ``` cd \
        mkdir hadoop 
        cd hadoop 
        mkdir bin 
    ```

###### Create Environment variables:

- Go to Environment Variable from Conttol Panel -> System -> Advanced System Settings.
  Go to "User variables for username" section.

- For Spark, click on New and enter Variable name as SPARK_HOME and Variable value as:
    ```
    C:\Spark\spark-3.0.1-bin-hadoop2.7 
    ```

- For Hadoop, click on New and enter Variable name as HADOOP_HOME and Variable value as:
    ``` 
    C:\hadoop 
    ```

- For Java, click on New and enter variable name as JAVA_HOME and Variable value as:
    ```
    C:\Program Files\Java\jre1.8.0_xxx 
    ```

- Select Path variable in User Variables section and click on Edit and add following 3 entries:
    ```%SPARK_HOME%\bin```
    ```%HADOOP_HOME%\bin```
    ```%JAVA_HOME%\bin```

- Save all the settings.

###### Launch and test Spark Installation

- Open command prompt, Navigate to C: and type:
    ``` 
    C:\Spark\spark-3.0.1-bin-hadoop2.7\bin\spark-shell
    ```

  or just type to test if Environment variables set earlier:
    ``` 
    spark-shell
    ```

- Following Scala prompt must launch.

    <img src="https://user-images.githubusercontent.com/8979477/93975395-678bb000-fd45-11ea-801a-c1d6ef702ff0.PNG" width="600" height="300">

- Navigate to http://localhost:4040/ on browser and you can see Apache Spark UI as follows:

    <img src="https://user-images.githubusercontent.com/8979477/93975399-68244680-fd45-11ea-8c07-2df313c5ae4e.PNG" width="600" height="300">

- Create a document with name "test" without any extension in C: and add some few lines with line  
  breaks, example from    
  here: [placeholder text](https://www.lipsum.com/).

- From Scala command prompt, type Following commands:

    ``` 
    val x =sc.textFile("test")
    x.take(2).foreach(println)
    ```
    
    <img src="https://user-images.githubusercontent.com/8979477/93975398-678bb000-fd45-11ea-9438-87d757665a81.PNG" width="500" height="200">

- Type ctrl-d to exit Spark shell.

##### Install Spark on Ubuntu\Linux\MacOS

- Refer instructions from [ubuntu](https://phoenixnap.com/kb/install-spark-on-ubuntu)
- Refer instructions from 
    [ubuntu-debian](https://computingforgeeks.com/how-to-install-apache-spark-on-ubuntu-debian/)
- Refer instructions from 
    [macOS](https://kevinvecmanis.io/python/pyspark/install/2019/05/31/Installing-Apache-Spark.html)

##### Install Apache Spark on Google Cloud
- Refer instructions from 
    [using Dataproc](https://codelabs.developers.google.com/codelabs/cloud-dataproc-starter/index.html?index=..%2F..index#0)
- Refer instructions to run Spark jobs on Google cloud 
    [google-cloud](https://cloud.google.com/dataproc/docs/tutorials/spark-scala)

#### How to configure Dataproc on Google Cloud and Apache Spark/Hadoop
- Refer Dataproc documentation [gc-dataproc](https://cloud.google.com/dataproc/docs)







