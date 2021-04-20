# Batch vs Stream Processing in Data Analyics and Machine Learning: an Overview in 2021

## Introduction

This is a note about batch processing vs stream processing, their technologies and toolstacks, especially for the use cases of data analytics and machine learning.

## Batch Processing

This is the traditional data analytics paradigm. Imagine the following scenario: a financial institute such as a bank collects data during the working hours, and performs analytics on the collected data during off hours. There is a clear gap between the data collection/loading step and the analytics step, and the data comes in batches, a bounded format.

## Batch Processing Technology Landscape

Easy mode. Most companies have used Spark, MapReduce, Hadoop, or a combination of these.

## Batch is cool, what about stream?

For the same financial institute, now they want to perform analytcis on the fly. Now they need stream processing.
Stream processing works with unbounded data instead of bounded data in batch processing. It is a type of data processing engine designed with infinite data in mind. One can argue batch processing is a special case of stream processing. Check [here](https://iwringer.wordpress.com/2015/08/03/patterns-for-streaming-realtime-analytics/) for a list of use cases scenarios/patterns where stream processing is needed

### Important Aspects of Stream Processing

1. Delivery Guarantees: in case of failures, the data will be processed 1. at least once, 2. at most once, or 3. exactly once. Exactly-once is the most desirable among the three, but it is harder to achieve, and there will be trade-off with performance.

2. Fault Tolerance: in case of failures, the process needs to be able to recover and start from where it left off. Usually this means it needs to do checkpointing on the state of processing and store it to some persistent storage.

3. State Management: stateless operations such as Map and FlatMap are purely functional and produce outputs solely based on the inputs. Stateful operations lke aggregation, on the other hand, requires additional side information stored in state.  A state in a data stream application is a data structure that preserves the history of past operations and influences the processing logic for future computations. Statee mangement of differnt frameworks usually fall on "a complexity continuum from
naive in-memory-only choice to a persistent state that can be queried and replicated".

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

2. Micro-batching, aka fast batching. This is really just batch processing, but applied to small enough batches to achieve a lower latency. Example: Apache Spark Streaming

Pros:

* Fault tolerane comes as free
* High throughput

Cons:

* not nutural streaming
* higher latency
* efficient state mangement is a challenge

### How to Choose?

Before we jump into the actuall framework and tech for streaming, we need to ask ourselves how do we want to choose our framework. Well, it all depends! Several things to consider:

1. Use cases. Does the use case require a very low latency? What is the minimum requirement for throughput? Is it just a simple use case that should use minimum tooling, or a very involved project?
2. Future considerations. Are there features/advanced features not needed now, but anticpated to be required in the future?
3. Existing tech stack. Do we have systems based on Hadoop filesystme/Kafka messaging queue/etc already?

### Stream Processing Tools

The following are cited from source [1].

#### Spark Streaming

Advantages:

* Supports Lambda architecture, comes free with Spark
* High throughput, good for many use cases where sub-latency is not required
* Fault tolerance by default due to micro-batch nature
* Simple to use higher level APIs
* Big community and aggressive improvements
* Exactly Once guarrantee

Disadvantages:

* Not true streaming, not suitable for low latency requirements
* Too many parameters to tune. Hard to get it right. Have written a post on my personal experience while tuning Spark Streaming
* Stateless by nature
* Lags behind Flink in many advanced features

#### Apache Kafka

Advantages:

* Very light weight library, good for microservices, IOT applications
* Does not need dedicated cluster
* Inherits all Kafka good characteristics
* Supports Stream joins, internally uses rocksDb for maintaining state.
* Exactly Once

Disadvantages:

* Tightly coupled with Kafka, can not use without Kafka in picture
* Quite new in infancy stage, yet to be tested in big companies
* Not for heavy lifting work like Spark Streaming,Flink.

#### Apache Flink

Advantages:

* Leader of innovation in open source Streaming landscape (Uber and Alibaba built their infrastructure with Flink)
* First True streaming framework with all advanced features like event time processing, watermarks, etc
* Low latency with high throughput, configurable according to requirements
* Auto-adjusting, not too many parameters to tune
* Exactly Once
* Getting widely accepted by big companies at scale like Uber,Alibaba.

Disadvantages:

* Little late in game, there was lack of adoption initially
* Community is not as big as Spark but growing at fast pace now
* No known adoption of the Flink Batch as of now, only popular for streaming.

## Reference and additional reading

[1] For a (slightly outdated) casual read, check out this Medium post: https://medium.com/@chandanbaranwal/spark-streaming-vs-flink-vs-storm-vs-kafka-streams-vs-samza-choose-your-stream-processing-91ea3f04675b

[2] For a more serious survey, read this IEEE paper: https://ieeexplore.ieee.org/document/8864052

[3] For understanding stream processing algorithms, check out this lecture from Stanford: https://youtu.be/lfJNJD7KkTg