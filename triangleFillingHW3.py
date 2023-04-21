import numpy
import cv2
import math

def ambientLight(surfaceAmbientReflectivity, lightIntensity):
    return surfaceAmbientReflectivity * lightIntensity  #equation 8.2


def normalize(vector):
    return vector / numpy.linalg.norm(vector, 2)


def diffuseLight(pointCoordinates, normalVector, pointColor, surfaceDiffuseReflectivity, lightPositions, lightIntensities):
    pointIntensity = numpy.array([0, 0, 0])

    lightSourcesTotal = len(lightIntensities)
    for lightSourceIdx in range(lightSourcesTotal):
        pointLightSourceVector = (lightPositions[lightSourceIdx] - pointCoordinates)  #L vector
        normalizedPointLightSourceVector = normalize(pointLightSourceVector)  #normalized L vector
        if numpy.dot(normalVector[0], normalizedPointLightSourceVector[0]) > 0:
            pointIntensity = pointIntensity + surfaceDiffuseReflectivity*lightIntensities[lightSourceIdx]*(numpy.dot(normalVector[0], normalizedPointLightSourceVector[0]))  #equation 8.5

    pointColor = pointColor + pointIntensity
    return pointColor


def specularLight(pointCoordinates, normalVector, pointColor, cameraCoordinates, surfaceSpecularReflectivity, phongFactor, lightPositions, lightIntensities):
    pointIntensity = numpy.array([0, 0, 0])

    pointCameraVector = (cameraCoordinates - pointCoordinates)  #V vector
    normalizedPointCameraVector = normalize(pointCameraVector)  #normalized V vector

    lightSourcesTotal = len(lightIntensities)
    for lightSourceIdx in range(lightSourcesTotal):
        pointLightSourceVector = (lightPositions[lightSourceIdx] - pointCoordinates)  #L vector
        normalizedPointLightSourceVector = normalize(pointLightSourceVector)  #normalized L vector
        reflectedPointLightSourceVector = 2*normalVector*numpy.dot(normalVector[0], normalizedPointLightSourceVector[0]) - normalizedPointLightSourceVector  #normalized R vector
        if numpy.dot(reflectedPointLightSourceVector[0], normalizedPointCameraVector[0]) > 0:
            pointIntensity = pointIntensity + surfaceSpecularReflectivity*lightIntensities[lightSourceIdx]*pow(numpy.dot(reflectedPointLightSourceVector[0], normalizedPointCameraVector[0]), phongFactor)  #equation 8.9

    pointColor = pointColor + pointIntensity
    return pointColor

def TriangleEdges(triangleVertices, triangleNormalVectors): # return the triangle edges as pairs of vertices, (edge = [vertex1, vertex2]) and corresponding edges.
    verticesTotal = 3
    triangleEdgesAndCorrespondingNormalVectors = []
    for startVertexIdx in range(verticesTotal):
        for endVertexIdx in range(startVertexIdx + 1, 3):
            triangleEdgesAndCorrespondingNormalVectors.append([[triangleVertices[startVertexIdx], triangleVertices[endVertexIdx]], triangleNormalVectors[startVertexIdx], triangleNormalVectors[endVertexIdx]])

    return triangleEdgesAndCorrespondingNormalVectors

def TriangleMinYCoord(trianglesVertices): # return triangle's minimum y coordinate.
    minYCoord = min(min(trianglesVertices[0][1], trianglesVertices[1][1]), trianglesVertices[2][1])
    return minYCoord

def TriangleMaxYCoord(trianglesVertices): # return triangle's maximum y coordinate.
    maxYCoord = max(max(trianglesVertices[0][1], trianglesVertices[1][1]), trianglesVertices[2][1])
    return maxYCoord

def EdgeMinXCoord(edge): # return edge's minimum x coordinate.
    edgeMinXCoord = min(edge[0][0], edge[1][0])
    return edgeMinXCoord

def EdgeMaxXCoord(edge): # return edges's maximum x coordinate.
    edgeMaxXCoord = max(edge[0][0], edge[1][0])
    return edgeMaxXCoord

def EdgeMinYCoord(edge): # return edges's minimum y coordinate.
    edgeMinYCoord = min(edge[0][1], edge[1][1])
    return edgeMinYCoord

def EdgeMaxYCoord(edge): # return edges's maximum y coordinate.
    edgeMaxYCoord = max(edge[0][1], edge[1][1])
    return edgeMaxYCoord

def IsHorizontal(edge): # check if edge is horizontal.
    if edge[0][1] == edge[1][1]:
        return True
    return False

def IsVertical(edge): # check if edge is vertical.
    if edge[0][0] == edge[1][0]:
        return True
    return False

def DeleteExpiredActiveEdges(activePointsAndCorrespondingEdges, yCoord): # delete expired edges based on their maximum y coordinate.
    activePointsAndCorrespondingEdges[:] = [pointAndCorrespEdge for pointAndCorrespEdge in activePointsAndCorrespondingEdges if EdgeMaxYCoord(pointAndCorrespEdge[1][0]) != yCoord]


def StartingActivePointsCorrespondingEdgesAndNormalVectors(triangleEdgesAndNormalVectors, triangleMinYCoord):
    edgesTotal = 3
    startingActivePointsCorrespondingEdgesAndNormalVectors = []

    for edgeIdx in range(edgesTotal):
        edge = triangleEdgesAndNormalVectors[edgeIdx][0]
        if (EdgeMinYCoord(edge) == triangleMinYCoord) and (IsHorizontal(edge) == False): # if edge is horizontal we don't append it.
            if EdgeMinYCoord(edge) == edge[0][1]:
                startingActivePointsCorrespondingEdgesAndNormalVectors.append([edge[0][0], triangleEdgesAndNormalVectors[edgeIdx]])
            else:
                startingActivePointsCorrespondingEdgesAndNormalVectors.append([edge[1][0], triangleEdgesAndNormalVectors[edgeIdx]])

    return startingActivePointsCorrespondingEdgesAndNormalVectors


def AppendNewActiveEdges(activePointsCorrespondingEdgesAndNormalVectors, triangleEdgesAndNormalVectors, yCoord): # append edges based on their minimum y coordinate.
    edgesTotal = 3
    for edgeIdx in range(edgesTotal):
        edge = triangleEdgesAndNormalVectors[edgeIdx][0]
        if (EdgeMinYCoord(edge) == yCoord) and (IsHorizontal(edge) == False): # if the edge is horizontal we don't append it.
            if EdgeMinYCoord(edge) == edge[0][1]:    # find edge's minimum x coordinate
                activePointsCorrespondingEdgesAndNormalVectors.append([edge[0][0], triangleEdgesAndNormalVectors[edgeIdx]])
            else:
                activePointsCorrespondingEdgesAndNormalVectors.append([edge[1][0], triangleEdgesAndNormalVectors[edgeIdx]])

def EdgeSlop(edge): # return the edge's slope.
    slope = (edge[1][1] - edge[0][1])/(edge[1][0] - edge[0][0])
    return slope


def UpdatePreviousActiveEdgesThatAreNotExpired(activePointsCorrespondingEdgesAndNormalVectors): # update active points based on the active edges they belong.
    for pointCorrespondingEdgeAndNormalVector in activePointsCorrespondingEdgesAndNormalVectors:
        edge = pointCorrespondingEdgeAndNormalVector[1][0]
        if IsVertical(edge) == False:
            pointCorrespondingEdgeAndNormalVector[0] = pointCorrespondingEdgeAndNormalVector[0] + 1/EdgeSlop(edge)


def updateActivePointsEdgesAndNormalVectors(activePointsCorrespondingEdgesAndNormalVectors, triangleEdgesAndNormalVectors, yCoord): # deletes expired edges, updates valid active edges, appends new active edges.
    DeleteExpiredActiveEdges(activePointsCorrespondingEdgesAndNormalVectors, yCoord)
    UpdatePreviousActiveEdgesThatAreNotExpired(activePointsCorrespondingEdgesAndNormalVectors)
    AppendNewActiveEdges(activePointsCorrespondingEdgesAndNormalVectors, triangleEdgesAndNormalVectors, yCoord)

def interpolate_color_and_normal_vector(endPointsCoordinate, endPointsColor, endPointsNormalVectors, interpolationPointCoordinate):
    interpolationPointColor = [None for _ in range(3)]
    interpolationPointNormalVector = [None for _ in range(3)]
    interpolationPointDistanceRatio = (endPointsCoordinate[1] - interpolationPointCoordinate) / (endPointsCoordinate[1] - endPointsCoordinate[0])

    for colorIndex in range(3):
        interpolationPointNormalVector[colorIndex] = interpolationPointDistanceRatio*endPointsNormalVectors[0][colorIndex] + (1 - interpolationPointDistanceRatio)* endPointsNormalVectors[1][colorIndex]
        interpolationPointColor[colorIndex] = interpolationPointDistanceRatio*endPointsColor[0][colorIndex] + (1 - interpolationPointDistanceRatio)*endPointsColor[1][colorIndex]

    return [interpolationPointColor, normalize(interpolationPointNormalVector)]

def EndPointsColorAndNormalVectors(edge, triangleVertices, triangleVerticesColor, triangleNormalVectors): # finds which triangle's vertex corresponds to each edge's end point so that we can
    verticesTotal = 3
    endPointsColor = [[None for _ in range(3)] for _ in range(2)]
    endPointsNormalVectors = [[None for _ in range(3)] for _ in range(2)]

    for vertexIdx in range(verticesTotal):
        if (triangleVertices[vertexIdx] == edge[0][0]) and (edge[0][0][1] < edge[0][1][1]):
            endPointsColor[0] = triangleVerticesColor[vertexIdx]
            endPointsNormalVectors[0] = triangleNormalVectors[vertexIdx]
        elif (triangleVertices[vertexIdx] == edge[0][0]) and (edge[0][0][1] > edge[0][1][1]):
            endPointsColor[1] = triangleVerticesColor[vertexIdx]
            endPointsNormalVectors[1] = triangleNormalVectors[vertexIdx]
        elif (triangleVertices[vertexIdx] == edge[0][1]) and (edge[0][0][1] < edge[0][1][1]):
            endPointsColor[1] = triangleVerticesColor[vertexIdx]
            endPointsNormalVectors[1] = triangleNormalVectors[vertexIdx]
        elif (triangleVertices[vertexIdx] == edge[0][1]) and (edge[0][0][1] > edge[0][1][1]):
            endPointsColor[0] = triangleVerticesColor[vertexIdx]
            endPointsNormalVectors[0] = triangleNormalVectors[vertexIdx]

    return [endPointsColor, endPointsNormalVectors]


def interpolate_color_and_normal_vector_last(endPointsCoordinate, endPointsColor, endPointsNormalVectors, interpolationPointCoordinate, eyeVector, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity, bcoords):
    interpolationPointColor = [None for _ in range(3)]
    interpolationPointNormalVector = [None for _ in range(3)]
    interpolationPointDistanceRatio = (endPointsCoordinate[1] - interpolationPointCoordinate) / (endPointsCoordinate[1] - endPointsCoordinate[0])
    if endPointsCoordinate[0] > interpolationPointCoordinate:
        interpolationPointDistanceRatio = 1

    for colorIndex in range(3):
        interpolationPointNormalVector[colorIndex] = interpolationPointDistanceRatio * endPointsNormalVectors[0][colorIndex] + (1 - interpolationPointDistanceRatio) * endPointsNormalVectors[1][colorIndex]
        interpolationPointColor[colorIndex] = interpolationPointDistanceRatio * endPointsColor[0][colorIndex] + (1 - interpolationPointDistanceRatio) * endPointsColor[1][colorIndex]

    interpolationPointNormalVector = normalize(interpolationPointNormalVector)

    interpolationPointColor = numpy.array(interpolationPointColor)
    interpolationPointColor = interpolationPointColor + ambientLight(ka, environmentalIntensity)
    interpolationPointColor = diffuseLight(numpy.transpose(bcoords), numpy.array([interpolationPointNormalVector]), interpolationPointColor, kd, lightPositions, lightIntensities)
    interpolationPointColor = specularLight(numpy.transpose(bcoords), numpy.array([interpolationPointNormalVector]), interpolationPointColor, eyeVector, ks, n, lightPositions, lightIntensities)

    return numpy.ndarray.tolist(interpolationPointColor)


def PhongColoring(xCoordIdx, yCoordIdx, activePointsAndCorrespondingEdges, triangleVertices, triangleVerticesColor, image, eyeVector, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity, bcoords, triangleNormalVectors): # implements gouraus coloring algorithm.
    firstActiveEdgeAndNormalVector  = activePointsAndCorrespondingEdges[0][1]
    secondActiveEdgeAndNormalVector = activePointsAndCorrespondingEdges[1][1]

    # finds left active edge and right active edge (because it is a triangle there will always be exactly 2 active edges).
    if firstActiveEdgeAndNormalVector[0][0][0] < secondActiveEdgeAndNormalVector[0][0][0] or (firstActiveEdgeAndNormalVector[0][0][0] == secondActiveEdgeAndNormalVector[0][0][0] and firstActiveEdgeAndNormalVector[0][1][0] < secondActiveEdgeAndNormalVector[0][1][0]):
        [leftActiveEdge, rightActiveEdge] = [firstActiveEdgeAndNormalVector, secondActiveEdgeAndNormalVector]
    else:
        [leftActiveEdge, rightActiveEdge] = [secondActiveEdgeAndNormalVector, firstActiveEdgeAndNormalVector]

    # calculates left active point's color with the color interpolation function based on the y coordinate.
    endPoints = [EdgeMinYCoord(leftActiveEdge[0]), EdgeMaxYCoord(leftActiveEdge[0])]
    [endPointsColor, endPointsNormalVectors] = EndPointsColorAndNormalVectors(leftActiveEdge, triangleVertices, triangleVerticesColor, triangleNormalVectors)
    [leftActiveEdgeInterpolatedPointColor, leftActiveEdgeInterpolatedNormalVector] = interpolate_color_and_normal_vector(endPoints, endPointsColor, endPointsNormalVectors, yCoordIdx)

    # calculates right active point's color with the color interpolation function based on the y coordinate.
    endPoints = [EdgeMinYCoord(rightActiveEdge[0]), EdgeMaxYCoord(rightActiveEdge[0])]
    [endPointsColor, endPointsNormalVectors] = EndPointsColorAndNormalVectors(rightActiveEdge, triangleVertices, triangleVerticesColor, triangleNormalVectors)
    [rightActiveEdgeInterpolatedPointColor, rightActiveEdgeInterpolatedNormalVector] = interpolate_color_and_normal_vector(endPoints, endPointsColor, endPointsNormalVectors, yCoordIdx)

    #  calculates interpolation point's color with the color interpolation function based on the x coordinate.
    firstActivePoint  = activePointsAndCorrespondingEdges[0][0]
    secondActivePoint = activePointsAndCorrespondingEdges[1][0]
    leftActivePointXCoord   = min(firstActivePoint, secondActivePoint)
    rightActivePointXCoord  = max(firstActivePoint, secondActivePoint)
    endPoints = [leftActivePointXCoord, rightActivePointXCoord]
    endPointsColor = [leftActiveEdgeInterpolatedPointColor, rightActiveEdgeInterpolatedPointColor]
    endPointsNormalVectors = [leftActiveEdgeInterpolatedNormalVector, rightActiveEdgeInterpolatedNormalVector]

    image[yCoordIdx][xCoordIdx] = interpolate_color_and_normal_vector_last(endPoints, endPointsColor, endPointsNormalVectors, xCoordIdx, eyeVector, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity, bcoords)


def shade_triangle_phoneHW3(image, triangleVertices, triangleVerticesColor, triangleNormalVectors, eyeVector, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity, bcoords): # color filling the triangle.
    triangleEdgesAndNormalVectors = TriangleEdges(triangleVertices, triangleNormalVectors) # finds triangle edges and corresponding normal vectors.
    triangleMinYCoord = TriangleMinYCoord(triangleVertices) # finds triangle's minimum y coordinate.
    triangleMaxYCoord = TriangleMaxYCoord(triangleVertices) # finds triangle's maximum y coordinate.

    activePointsCorrespondingEdgesAndNormalVectors = StartingActivePointsCorrespondingEdgesAndNormalVectors(triangleEdgesAndNormalVectors, triangleMinYCoord) # finds starting active points, corresponding edges and normal vectors.

    for yCoordIdx in range(int(triangleMinYCoord), int(triangleMaxYCoord)): # from triangle's minimum scan line to triangle's maximum scan line.

        leftActivePoint  = min(activePointsCorrespondingEdgesAndNormalVectors[0][0], activePointsCorrespondingEdgesAndNormalVectors[1][0]) # find left active point.
        rightActivePoint = max(activePointsCorrespondingEdgesAndNormalVectors[0][0], activePointsCorrespondingEdgesAndNormalVectors[1][0]) # find right active point.

        for xCoordIdx in range(int(leftActivePoint), int(rightActivePoint)): # from the left active point to the right active point minus one.
            PhongColoring(xCoordIdx, yCoordIdx, activePointsCorrespondingEdgesAndNormalVectors, triangleVertices, triangleVerticesColor, image, eyeVector, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity, bcoords, triangleNormalVectors)
        updateActivePointsEdgesAndNormalVectors(activePointsCorrespondingEdgesAndNormalVectors, triangleEdgesAndNormalVectors, yCoordIdx + 1) # update active points and corresponding edges.

    return image
