# Batch vs Stream Processing in Data Analyics and Machine Learning: an Overview in 2021

## Introduction

This is a note about batch processing vs stream processing, their technologies and toolstacks, especially for the use cases of data analytics and machine learning.

## Batch Processing

## Batch Processing Technology Landscape

Easy mode. Most companies have used Spark, MapReduce, Hadoop, or a combination of these.

## Batch is cool, what about stream?

Stream processing, as oopposed to batch processing, works with unbounded data instead of bounded data. It is a type of data processing engine designed with infinite data in mind. One can argue batch processing is a special case of stream processing. 

### Important Aspects of Stream Processing

1. Delivery Guarantees: in case of failures, the data will be processed 1. at least once, 2. at most once, or 3. exactly once. Exactly-once is the most desirable among the three, but it is harder to achieve, and there will be trade-off with performance.

2. Fault Tolerance: in case of failures, the process needs to be able to recover and start from where it left off. Usually this means it needs to do checkpointing on the state of processing and store it to some persistent storage.

3. State Management

4. Performance: latency (as low as possible), throughput (as high as possible), and scalability.

5. Advanced features: 1. event time processing, 2. watermakrs, and 3. windowing.

6. Maturity: is the framework battle tested at scale? How good is the community support? etc

### Two Types of Stream Processing and Their Pros/Cons

1. Natrual streaming, aka native streaming. This is the true streaming, a contiuous process that runs forever to process incoming data. Examples: Apache Kafka Stream, Apache Flink.

Pros:

* feels natural for streaming
* low latency
* state mangement is easy

Cons:

* harder to achieve fault tolerance

2. Micro-batching, aka fast baatching. This is really just batch processing, but applied to small enough batches to achieve a lower latency. Example: Apache Spark Streaming

Pros:

* Fault tolerane comes as free
* High throughput

Cons:

* not nutural streaming
* higher latency
* efficient state mangement is a challenge

### How to Choose?

Before we jump into the actuall framework and tech for streaming, we need to ask ourselves how do we want to choose our framework. Well, it all depends! Several things to consider:

1. Use cases.
2. Future considerations.
3. Existing tech stack.


### Stream Processing Tools

#### Spark Streaming


#### Apache Kafka


#### Apache Flink


#### AWS Kinesis


## Reference
