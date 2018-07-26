'''Feature selector local server setup'''

from concurrent import futures
import time
import sys

import grpc

import featureselector_pb2
import featureselector_pb2_grpc

import numpy as np
import sel

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class FeatureSelector(featureselector_pb2_grpc.FeatureSelectorServicer):

    def fsLassoCV(self, request, context):
        data = []
        feat_list = []
        for col in request.covariates:
            data += [col.entries]
            feat_list += [col.name]

        data = np.transpose(np.array(data))
        output = sel.fs_lasso_cv(data,request.response.entries,feat_list)

        return featureselector_pb2.stringList(messages=output)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    featureselector_pb2_grpc.add_FeatureSelectorServicer_to_server(FeatureSelector(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
