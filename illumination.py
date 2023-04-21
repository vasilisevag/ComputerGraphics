import numpy
from demo import project_cam_lookat, rastersize
from triangleFilling import *
from triangleFillingHW3 import shade_triangle_phoneHW3

def shade_triangle_gouraudHW3(image, triangleVertices, triangleVerticesColor):  #exactly the same as shade_triangle_gouraud from project 1 except that it returns the modified image
    triangleEdges = TriangleEdges(triangleVertices)
    triangleMinYCoord = TriangleMinYCoord(triangleVertices)
    triangleMaxYCoord = TriangleMaxYCoord(triangleVertices)

    activePointsAndCorrespondingEdges = StartingActivePointsAndCorrespondingEdges(triangleEdges, triangleMinYCoord)

    for yCoordIdx in range(int(triangleMinYCoord), int(triangleMaxYCoord)):
        leftActivePoint = min(activePointsAndCorrespondingEdges[0][0], activePointsAndCorrespondingEdges[1][0])
        rightActivePoint = max(activePointsAndCorrespondingEdges[0][0], activePointsAndCorrespondingEdges[1][0])

        for xCoordIdx in range(int(leftActivePoint), int(rightActivePoint)):
            GouraudColoring(xCoordIdx, yCoordIdx, activePointsAndCorrespondingEdges, triangleVertices, triangleVerticesColor, image)

        updateActivePointsAndEdges(activePointsAndCorrespondingEdges, triangleEdges, yCoordIdx + 1)

    return image


def normalize(vector):  # normalize input vector using the euclidean norm
    return vector / numpy.linalg.norm(vector, 2)


def normalizeImage(image):  # normalize input image in range [0, 1] by dividing every pixel with the maximum value pixel
    imageHeight = len(image)
    imageWidth = len(image[0])
    colorChannelsTotal = len(image[0][0])

    maxValue = max(max(max(image)))

    for heightIdx in range(imageHeight):
        for widthIdx in range(imageWidth):
            for colorIdx in range(colorChannelsTotal):
                image[heightIdx][widthIdx][colorIdx] = image[heightIdx][widthIdx][colorIdx] / maxValue

    return image


def ambientLight(surfaceAmbientReflectivity, lightIntensity):
    return surfaceAmbientReflectivity * lightIntensity  #equation 8.2


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


def calculateNormals(vertices, faceIndices):
    verticesTotal = vertices.shape[1]
    normalVectors = numpy.zeros([3, verticesTotal])  #a zero initialized array for the normal vectors of each vertex
    vertexTrianglesCount = numpy.zeros([verticesTotal])  #a zero initialized array for the number of triangles each vertex belongs to

    trianglesTotal = faceIndices.shape[1]
    for triangleIdx in range(trianglesTotal):
        triangleVertices = numpy.zeros([3, 3])

        for vertexIdx in range(3):
            triangleVertices[:, vertexIdx] = vertices[:, faceIndices[vertexIdx, triangleIdx]]

        triangleEdgeABVector = triangleVertices[:, 1] - triangleVertices[:, 0]  #triangle = (A, B, C), calculate vector AB and BC, then normalize their cross product
        triangleEdgeBCVector = triangleVertices[:, 2] - triangleVertices[:, 1]
        normalizedNormalVector = normalize(numpy.cross(triangleEdgeABVector, triangleEdgeBCVector))

        for vertexIdx in range(3):
            vertexTrianglesCount[faceIndices[vertexIdx, triangleIdx]] = vertexTrianglesCount[faceIndices[vertexIdx, triangleIdx]] + 1  #update the number of triangles the vertex belongs to
            normalVectors[:, faceIndices[vertexIdx, triangleIdx]] = normalVectors[:, faceIndices[vertexIdx, triangleIdx]] + normalizedNormalVector  #update vertex normal vector

    for vertexIdx in range(verticesTotal):
        normalVectors[:, vertexIdx] = normalize(normalVectors[:, vertexIdx] / vertexTrianglesCount[vertexIdx])   # normalize normal vector based on the number of triangles its vertex belongs to

    return normalVectors


def renderObject(shader, focalLength, eyeVector, lookatVector, upVector, backgroundColor, imageHeight, imageWidth, ccdHeight, ccdWidth, vertices, verticesColors, faceIndices, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity):
    verticesNormalVectors = calculateNormals(vertices, faceIndices)  #get normal vectors
    [cameraPoints, vertexDepth] = project_cam_lookat(focalLength, eyeVector, lookatVector, upVector, vertices)  #get camera points and vertex depths
    cameraPixels = rastersize(cameraPoints, imageHeight, imageWidth, ccdHeight, ccdWidth)  #get pixel points

    trianglesTotal = faceIndices.shape[1]  #sort the triangles based on their depth
    triangleDepth = [0.0] * trianglesTotal
    for triangleIdx in range(trianglesTotal):
        triangleDepth[triangleIdx] = (vertexDepth[faceIndices[0][triangleIdx]] + vertexDepth[faceIndices[1][triangleIdx]] + vertexDepth[faceIndices[2][triangleIdx]]) / 3

    triangleVertices = numpy.ndarray.tolist(numpy.transpose(faceIndices))
    triangleVertices = [triangle for _, triangle in sorted(zip(triangleDepth, triangleVertices))]
    triangleVertices = [triangle for triangle in reversed(triangleVertices)]

    image = [[backgroundColor for _ in range(imageHeight)] for _ in range(imageWidth)]
    for triangleIdx in range(trianglesTotal):  # for each triangle
        vertsp = numpy.zeros([2, 3])  #get its vertices coordinates
        for vertexIdx in range(3):
            vertsp[:, vertexIdx] = cameraPixels[:, triangleVertices[triangleIdx][vertexIdx]]

        vertsn = numpy.zeros([3, 3])  #get its vertices normal vectors
        for vertexIdx in range(3):
            vertsn[:, vertexIdx] = verticesNormalVectors[:, triangleVertices[triangleIdx][vertexIdx]]

        vertsc = numpy.zeros([3, 3])  #get its vertices colors
        for vertexIdx in range(3):
            vertsc[:, vertexIdx] = verticesColors[:, triangleVertices[triangleIdx][vertexIdx]]

        bcoords = numpy.zeros([1, 3])  #get the centroid of the triangle
        for vertexIdx in range(3):
            bcoords = bcoords + vertices[:, triangleVertices[triangleIdx][vertexIdx]]
        bcoords = bcoords / 3
        bcoords = numpy.transpose(bcoords)

        if shader == "gouraud":
            image = shadeGouraud(vertsp, vertsn, vertsc, bcoords, eyeVector, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity, image)  #shade it using Gouraud
        elif shader == "phong":
            image = shadePhong(vertsp, vertsn, vertsc, bcoords, eyeVector, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity, image)  # shade it using Phong

    return normalizeImage(image)


def shadeGouraud(vertsp, vertsn, vertsc, bcoords, eyeVector, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity, image):
    for colorIdx in range(3):  #update the color of each triangle's vertex using the illumination models
        vertsc[:, colorIdx] = vertsc[:, colorIdx] + ambientLight(ka, environmentalIntensity)
        vertsc[:, colorIdx] = diffuseLight(numpy.transpose(bcoords), numpy.array([vertsn[:, colorIdx]]), vertsc[:, colorIdx], kd, lightPositions, lightIntensities)
        vertsc[:, colorIdx] = specularLight(numpy.transpose(bcoords), numpy.array([vertsn[:, colorIdx]]), vertsc[:, colorIdx], eyeVector, ks, n, lightPositions, lightIntensities)

    return shade_triangle_gouraudHW3(image, numpy.ndarray.tolist(numpy.transpose(vertsp)), numpy.ndarray.tolist(numpy.transpose(vertsc)))  #call shade_triangle_gouraudHW3


def shadePhong(vertsp, vertsn, vertsc, bcoords, eyeVector, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity, image):
    triangleVertices = numpy.ndarray.tolist(numpy.transpose(vertsp))  #just do some transformations from numpy arrays to lists (for the sake of backwards compatibility)
    triangleVerticesColor = numpy.ndarray.tolist(numpy.transpose(vertsc))
    triangleNormalVectors = numpy.ndarray.tolist(numpy.transpose(vertsn))
    #  shade_triangle_phoneHW3 is in the triangleFillingHW3 file (as I said in the report as well, triangleFillingHW3 file is just an updated version of the project's 1 file that can handle, in addition to the previous one, normal vector interpolations
    return shade_triangle_phoneHW3(image, triangleVertices, triangleVerticesColor, triangleNormalVectors, eyeVector, ka, kd, ks, n, lightPositions, lightIntensities, environmentalIntensity, bcoords)


def LoadParameters(filename):  #just loads the parameters from the given file
    parameters = numpy.load(filename, allow_pickle=True) # loads file's parameters.
    parameters = numpy.ndarray.tolist(parameters) # convert numpy array to list.

    return [parameters["verts"], parameters["vertex_colors"], parameters["face_indices"], parameters["depth"], parameters["cam_eye"], parameters["cam_up"], parameters["cam_lookat"], parameters["ka"], parameters["kd"], parameters["ks"], parameters["n"], parameters["light_positions"], parameters["light_intensities"], parameters["Ia"], parameters["M"], parameters["N"], parameters["W"], parameters["H"], parameters["bg_color"]]


def demoRun():
    [vertices, verticesColors, faceIndices, depth, cam_eye, cam_up, cam_lookat, ka, kd, ks, n, lightPositions, lightIntensities, Ia, M, N, W, H, bg_color] = LoadParameters("h3.npy")
    vertices = numpy.transpose(vertices)  #transforming some parameters by transposing them
    verticesColors = numpy.transpose(verticesColors)  #Their shape was in the form [N, 3] while the functions we created expect a [3, n] shape
    faceIndices = numpy.transpose(faceIndices)

    #  get all the requested images
    gouraudImageEnvironmentalLight = renderObject("gouraud", 70, cam_eye, cam_lookat, cam_up, [0, 0, 0], M, N, H, W, vertices, verticesColors, faceIndices, ka, 0, 0, n, lightPositions, lightIntensities, Ia)
    RGB2BGR(gouraudImageEnvironmentalLight)
    npyGouraudImage = numpy.array(gouraudImageEnvironmentalLight)

    phongImageEnvironmentalLight = renderObject("phong", 70, cam_eye, cam_lookat, cam_up, [0, 0, 0], M, N, H, W, vertices, verticesColors, faceIndices, ka, 0, 0, 0, lightPositions, lightIntensities, Ia)
    RGB2BGR(phongImageEnvironmentalLight)
    npyPhongImage = numpy.array(phongImageEnvironmentalLight)  # convert image (list of lists) to a numpy array.

    gouraudImageDiffusionLight = renderObject("gouraud", 70, cam_eye, cam_lookat, cam_up, [0, 0, 0], M, N, H, W, vertices, verticesColors, faceIndices, 0, kd, 0, n, lightPositions, lightIntensities, Ia)
    RGB2BGR(gouraudImageDiffusionLight)
    npyGouraudImage = numpy.array(gouraudImageDiffusionLight)

    phongImageDiffusionLight = renderObject("phong", 70, cam_eye, cam_lookat, cam_up, [0, 0, 0], M, N, H, W, vertices, verticesColors, faceIndices, 0, kd, 0, n, lightPositions, lightIntensities, Ia)
    RGB2BGR(phongImageDiffusionLight)
    npyPhongImage = numpy.array(phongImageDiffusionLight)  # convert image (list of lists) to a numpy array.

    gouraudImageReflectionLight = renderObject("gouraud", 70, cam_eye, cam_lookat, cam_up, [0, 0, 0], M, N, H, W, vertices, verticesColors, faceIndices, 0, 0, ks, n, lightPositions, lightIntensities, Ia)
    RGB2BGR(gouraudImageReflectionLight)
    npyGouraudImage = numpy.array(gouraudImageReflectionLight)

    phongImageReflectionLight = renderObject("phong", 70, cam_eye, cam_lookat, cam_up, [0, 0, 0], M, N, H, W, vertices, verticesColors, faceIndices, 0, 0, ks, n, lightPositions, lightIntensities, Ia)
    RGB2BGR(phongImageReflectionLight)
    npyPhongImage = numpy.array(phongImageReflectionLight)  # convert image (list of lists) to a numpy array.

    gouraudImageFullModel = renderObject("gouraud", 70, cam_eye, cam_lookat, cam_up, [0, 0, 0], M, N, H, W, vertices, verticesColors, faceIndices, ka, kd, ks, n, lightPositions, lightIntensities, Ia)
    RGB2BGR(gouraudImageFullModel)
    npyGouraudImage = numpy.array(gouraudImageFullModel)

    phongImageFullModel = renderObject("phong", 70, cam_eye, cam_lookat, cam_up, [0, 0, 0], M, N, H, W, vertices, verticesColors, faceIndices, ka, kd, ks, n, lightPositions, lightIntensities, Ia)
    RGB2BGR(phongImageFullModel)
    npyPhongImage = numpy.array(phongImageFullModel)  # convert image (list of lists) to a numpy array.


    # concatenate a big image consisting of all the previous images and show it
    concatenatedImageGouraud = numpy.concatenate((gouraudImageEnvironmentalLight, gouraudImageDiffusionLight, gouraudImageReflectionLight, gouraudImageFullModel), axis=1)
    concatenatedImagePhong = numpy.concatenate((phongImageEnvironmentalLight, phongImageDiffusionLight, phongImageReflectionLight, phongImageFullModel), axis=1)
    concatenatedImage = numpy.concatenate((concatenatedImageGouraud, concatenatedImagePhong), axis=0)
    cv2.imshow("TOP: GOURAUD --- BOTTOM: PHONG", concatenatedImage)  # show concatenated image.
    cv2.waitKey()


demoRun()
