import numpy
import cv2

def interpolate_color(endPointsCoordinate, endPointsColor, interpolationPointCoordinate): # return point color with linear interpolation.
    interpolationPointColor = [None for _ in range(3)]
    interpolationPointDistanceRatio = (endPointsCoordinate[1] - interpolationPointCoordinate) / (endPointsCoordinate[1] - endPointsCoordinate[0])

    for colorIndex in range(3):
        interpolationPointColor[colorIndex] = interpolationPointDistanceRatio*endPointsColor[0][colorIndex] + (1 - interpolationPointDistanceRatio)*endPointsColor[1][colorIndex]

    return interpolationPointColor

def TriangleEdges(triangleVertices): # return the triangle edges as pairs of vertices, (edge = [vertex1, vertex2]).
    verticesTotal = 3
    triangleEdges = []
    for startVertexIdx in range(3):
        for endVertexIdx in range(startVertexIdx + 1, 3):
            triangleEdges.append([triangleVertices[startVertexIdx], triangleVertices[endVertexIdx]])

    return triangleEdges

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
    activePointsAndCorrespondingEdges[:] = [pointAndCorrespEdge for pointAndCorrespEdge in activePointsAndCorrespondingEdges if EdgeMaxYCoord(pointAndCorrespEdge[1]) != yCoord]

def AppendNewActiveEdges(activePointsAndCorrespondingEdges, triangleEdges, yCoord): # append edges based on their minimum y coordinate.
    edgesTotal = 3
    for edgeIdx in range(edgesTotal):
        edge = triangleEdges[edgeIdx]
        if (EdgeMinYCoord(edge) == yCoord) and (IsHorizontal(edge) == False): # if the edge is horizontal we don't append it.
            if EdgeMinYCoord(edge) == edge[0][1]:    # find edge's minimum x coordinate
                activePointsAndCorrespondingEdges.append([edge[0][0], edge])
            else:
                activePointsAndCorrespondingEdges.append([edge[1][0], edge])

def EdgeSlop(edge): # return the edge's slope.
    slope = (edge[1][1] - edge[0][1])/(edge[1][0] - edge[0][0])
    return slope

def UpdatePreviousActiveEdgesThatAreNotExpired(activePointsAndCorrespondingEdges): # update active points based on the active edges they belong.
    for pointAndCorrespondingEdge in activePointsAndCorrespondingEdges:
        edge = pointAndCorrespondingEdge[1]
        if IsVertical(edge) == False:
            pointAndCorrespondingEdge[0] = pointAndCorrespondingEdge[0] + 1/EdgeSlop(edge)

def updateActivePointsAndEdges(activePointsAndCorrespondingEdges, triangleEdges, yCoord): # deletes expired edges, updates valid active edges, appends new active edges.
    DeleteExpiredActiveEdges(activePointsAndCorrespondingEdges, yCoord)
    UpdatePreviousActiveEdgesThatAreNotExpired(activePointsAndCorrespondingEdges)
    AppendNewActiveEdges(activePointsAndCorrespondingEdges, triangleEdges, yCoord)

def FlatColoring(xCoordIdx, yCoordIdx, triangleVerticesColor, image): # calculates and returns a point's color based only on the colors of the triangle's vertices
    colorChannelTotal = 3                                             # it belongs to. Independent from it's distance between triangle's vertices.
    verticesTotal = 3
    colorChannel = [0.0] * colorChannelTotal

    for colorIdx in range(colorChannelTotal):
        for vertexIdx in range(verticesTotal):
            colorChannel[colorIdx] = colorChannel[colorIdx] + triangleVerticesColor[vertexIdx][colorIdx]
        colorChannel[colorIdx] = colorChannel[colorIdx] / 3

    image[yCoordIdx][xCoordIdx] = colorChannel

def EndPointsColor(edge, triangleVertices, triangleVerticesColor): # finds which triangle's vertex corresponds to each edge's end point so that we can
    verticesTotal = 3                                              # calculate the colors of the edge's end points
    endPointsColor = [[None for _ in range(3)] for _ in range(2)]

    for vertexIdx in range(verticesTotal):
        if (triangleVertices[vertexIdx] == edge[0]) and (edge[0][1] < edge[1][1]):
            endPointsColor[0] = triangleVerticesColor[vertexIdx]
        elif (triangleVertices[vertexIdx] == edge[0]) and (edge[0][1] > edge[1][1]):
            endPointsColor[1] = triangleVerticesColor[vertexIdx]
        elif (triangleVertices[vertexIdx] == edge[1]) and (edge[0][1] < edge[1][1]):
            endPointsColor[1] = triangleVerticesColor[vertexIdx]
        elif (triangleVertices[vertexIdx] == edge[1]) and (edge[0][1] > edge[1][1]):
            endPointsColor[0] = triangleVerticesColor[vertexIdx]

    return endPointsColor

def GouraudColoring(xCoordIdx, yCoordIdx, activePointsAndCorrespondingEdges, triangleVertices, triangleVerticesColor, image): # implements gouraus coloring algorithm.
    colorChannelTotal = 3
    leftActiveEdge  = None
    rightActiveEdge = None
    firstActiveEdge  = activePointsAndCorrespondingEdges[0][1]
    secondActiveEdge = activePointsAndCorrespondingEdges[1][1]

    # finds left active edge and right active edge (because it is a triangle there will always be exactly 2 active edges).
    if firstActiveEdge[0][0] < secondActiveEdge[0][0] or (firstActiveEdge[0][0] == secondActiveEdge[0][0] and firstActiveEdge[1][0] < secondActiveEdge[1][0]):
        [leftActiveEdge, rightActiveEdge] = [firstActiveEdge, secondActiveEdge]
    else:
        [leftActiveEdge, rightActiveEdge] = [secondActiveEdge, firstActiveEdge]

    # calculates left active point's color with the color interpolation function based on the y coordinate.
    endPoints = [EdgeMinYCoord(leftActiveEdge), EdgeMaxYCoord(leftActiveEdge)]
    endPointsColor = EndPointsColor(leftActiveEdge, triangleVertices, triangleVerticesColor)
    leftActiveEdgeInterpolatedPointColor = interpolate_color(endPoints, endPointsColor, yCoordIdx)

    # calculates right active point's color with the color interpolation function based on the y coordinate.
    endPoints = [EdgeMinYCoord(rightActiveEdge), EdgeMaxYCoord(rightActiveEdge)]
    endPointsColor = EndPointsColor(rightActiveEdge, triangleVertices, triangleVerticesColor)
    rightActiveEdgeInterpolatedPointColor = interpolate_color(endPoints, endPointsColor, yCoordIdx)

    #  calculates interpolation point's color with the color interpolation function based on the x coordinate.
    firstActivePoint  = activePointsAndCorrespondingEdges[0][0]
    secondActivePoint = activePointsAndCorrespondingEdges[1][0]
    leftActivePointXCoord   = min(firstActivePoint, secondActivePoint)
    rightActivePointXCoord  = max(firstActivePoint, secondActivePoint)
    endPoints = [leftActivePointXCoord, rightActivePointXCoord]
    endPointsColor = [leftActiveEdgeInterpolatedPointColor, rightActiveEdgeInterpolatedPointColor]

    image[yCoordIdx][xCoordIdx] = interpolate_color(endPoints, endPointsColor, xCoordIdx)

def StartingActivePointsAndCorrespondingEdges(triangleEdges, triangleMinYCoord): # finds and returns the staring active points and corresponding edges based on the
    edgesTotal = 3                                                               # triangle's minimum y coordinate.
    startingActivePointsAndCorrespondingEdges = []

    for edgeIdx in range(edgesTotal):
        edge = triangleEdges[edgeIdx]
        if (EdgeMinYCoord(edge) == triangleMinYCoord) and (IsHorizontal(edge) == False): # if edge is horizontal we don't append it.
            if EdgeMinYCoord(edge) == edge[0][1]:
                startingActivePointsAndCorrespondingEdges.append([edge[0][0], edge])
            else:
                startingActivePointsAndCorrespondingEdges.append([edge[1][0], edge])

    return startingActivePointsAndCorrespondingEdges

def shade_triangle(image, triangleVertices, triangleVerticesColor, shade_t): # color filling the triangle.
    verticesTotal = 3
    height = len(image)
    width = len(image[0])

    triangleEdges = TriangleEdges(triangleVertices) # finds triangle edges.
    triangleMinYCoord = TriangleMinYCoord(triangleVertices) # finds triangle's minimum y coordinate.
    triangleMaxYCoord = TriangleMaxYCoord(triangleVertices) # finds triangle's maximum y coordinate.

    activePointsAndCorrespondingEdges = StartingActivePointsAndCorrespondingEdges(triangleEdges, triangleMinYCoord) # finds starting active points and corresponding edges.

    for yCoordIdx in range(int(triangleMinYCoord), int(triangleMaxYCoord)): # from triangle's minimum scan line to triangle's maximum scan line.
        leftActivePoint  = min(activePointsAndCorrespondingEdges[0][0], activePointsAndCorrespondingEdges[1][0]) # find left active point.
        rightActivePoint = max(activePointsAndCorrespondingEdges[0][0], activePointsAndCorrespondingEdges[1][0]) # find right active point.

        for xCoordIdx in range(int(leftActivePoint), int(rightActivePoint)): # from the left active point to the right active point minus one.
            if shade_t == "flat": # color that point based on the shade_t string.
                FlatColoring(xCoordIdx, yCoordIdx, triangleVerticesColor, image)
            elif shade_t == "gouraud":
                GouraudColoring(xCoordIdx, yCoordIdx, activePointsAndCorrespondingEdges, triangleVertices, triangleVerticesColor, image)
        updateActivePointsAndEdges(activePointsAndCorrespondingEdges, triangleEdges, yCoordIdx + 1) # update active points and corresponding edges.

def render(imageVertices, triangleVertices, imageVerticesColor, vertexDepth, shade_t):
    image = [[[1 for _ in range(3)] for _ in range(512)] for _ in range(512)] # create image (as a list of lists).
    trianglesTotal = len(triangleVertices)
    triangleDepth = [0.0] * trianglesTotal

    for triangleIdx in range(trianglesTotal): # find the depth of each triangle.
        triangleDepth[triangleIdx] = (vertexDepth[triangleVertices[triangleIdx][0]] + vertexDepth[triangleVertices[triangleIdx][1]] + vertexDepth[triangleVertices[triangleIdx][2]]) / 3

    # sort the triangles based on their depth.
    triangleVertices = [triangle for _,triangle in sorted(zip(triangleDepth, triangleVertices))]
    triangleVertices = [triangle for triangle in reversed(triangleVertices)]

    # shade each triangle.
    for triangle in triangleVertices:
        shade_triangle(image, [imageVertices[triangle[0]], imageVertices[triangle[1]], imageVertices[triangle[2]]], [imageVerticesColor[triangle[0]], imageVerticesColor[triangle[1]], imageVerticesColor[triangle[2]]], shade_t)

    return image

def LoadParameters(filename):
    parameters = numpy.load(filename, allow_pickle=True) # loads file's parameters.
    parameters = numpy.ndarray.tolist(parameters) # convert numpy array to list.

    # separate and return the parameters.
    return [numpy.ndarray.tolist(parameters["verts2d"]), numpy.ndarray.tolist(parameters["vcolors"]), numpy.ndarray.tolist(parameters["faces"]), numpy.ndarray.tolist(parameters["depth"])]

def RGB2BGR(rgbImage):
    for i in range(512):
        for j in range(512):
            rgbImage[i][j][2], rgbImage[i][j][0] = rgbImage[i][j][0], rgbImage[i][j][2]

def RunDemo():
    [verts2d, vcolors, faces, depth] = LoadParameters("hw1.npy") # load parameters.

    flatImage = render(verts2d, faces, vcolors, depth, "flat") # render flat image.
    RGB2BGR(flatImage)
    npyFlatImage = numpy.array(flatImage) # convert image (list of lists) to a numpy array.

    gouraudImage = render(verts2d, faces, vcolors, depth, "gouraud")  # render gouraud image.
    RGB2BGR(gouraudImage)
    npyGouraudImage = numpy.array(gouraudImage) # convert image (list of lists) to a numpy array

    concatenatedImage = numpy.concatenate((flatImage, gouraudImage), axis = 1) # concatenate the two images horizontally
    cv2.imshow("image", concatenatedImage) # show concatenated image.
    cv2.waitKey()

#RunDemo() # runs both demos. It takes about 5 seconds to render and show the images.
