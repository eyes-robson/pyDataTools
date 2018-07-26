'''Client for a simple gRPC example using the dataSci folders'''

from __future__ import print_function
from concurrent import futures

import grpc

import featureselector_pb2
import featureselector_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = featureselector_pb2_grpc.FeatureSelectorStub(channel)

    # create the data to test
    X1 = featureselector_pb2.numericArray.Column(name='first', entries=[2,4,7,6,9,14,19])
    X2 = featureselector_pb2.numericArray.Column(name='second', entries=[3,2,1,2,1,5,8])
    X3 = featureselector_pb2.numericArray.Column(name='third', entries=[0,1,2,5,4,6,5])
    X4 = featureselector_pb2.numericArray.Column(name='fourth', entries=[0,0,0,1,1,0,0])
    y = featureselector_pb2.numericArray.Column(name='response', entries=[1,2,3,4,6,7,10])

    data = featureselector_pb2.numericArray(covariates = [X1, X2, X3, X4], response=y)
    response = stub.fsLassoCV(data)
    print("FeatureSelector client received: " + str([x for x in response.messages]))

if __name__ == '__main__':
    run()
