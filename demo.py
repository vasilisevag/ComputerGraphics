import numpy
import math
import cv2
from triangleFilling import *

def affine_transform(cp, theta, u, t):
    if cp.ndim == 1:
        cq = numpy.copy(cp)
        if theta is not None:
            rtheta = math.radians(theta)
            cq[0] = ((1 - math.cos(rtheta))*pow(u[0], 2) + math.cos(rtheta))*cp[0] + ((1 - math.cos(rtheta))*u[0]*u[1] - math.sin(rtheta)*u[2])*cp[1] + ((1 - math.cos(rtheta))*u[0]*u[2] + math.sin(rtheta)*u[1])*cp[2]
            cq[1] = ((1 - math.cos(rtheta))*u[1]*u[0] + math.sin(rtheta)*u[2])*cp[0] + ((1 - math.cos(rtheta))*pow(u[1], 2) + math.cos(rtheta))*cp[1] + ((1 - math.cos(rtheta))*u[1]*u[2] - math.sin(rtheta)*u[0])*cp[2]
            cq[2] = ((1 - math.cos(rtheta))*u[2]*u[0] - math.sin(rtheta)*u[1])*cp[0] + ((1 - math.cos(rtheta))*u[2]*u[1] + math.sin(rtheta)*u[0])*cp[1] + ((1 - math.cos(rtheta))*pow(u[2], 2) + math.cos(rtheta))*cp[2]
        if t is not None:
            for dimIdx in range(3):
                cq[dimIdx] = cq[dimIdx] + t[dimIdx]
        return cq

    else:
        pointsTotal = cp.shape[1]
        cq = numpy.copy(cp)
        if theta is not None:
            rtheta = math.radians(theta)
            for pointIdx in range(pointsTotal):
                cq[0][pointIdx] = ((1 - math.cos(rtheta)) * pow(u[0], 2) + math.cos(rtheta))*cp[0][pointIdx] + ((1 - math.cos(rtheta)) * u[0] * u[1] - math.sin(rtheta) * u[2])*cp[1][pointIdx] + ((1 - math.cos(rtheta)) * u[0] * u[2] + math.sin(rtheta) * u[1])*cp[2][pointIdx]
                cq[1][pointIdx] = ((1 - math.cos(rtheta)) * u[1] * u[0] + math.sin(rtheta) * u[2])*cp[0][pointIdx] + ((1 - math.cos(rtheta)) * pow(u[1], 2) + math.cos(rtheta))*cp[1][pointIdx] + ((1 - math.cos(rtheta)) * u[1] * u[2] - math.sin(rtheta) * u[0])*cp[2][pointIdx]
                cq[2][pointIdx] = ((1 - math.cos(rtheta)) * u[2] * u[0] - math.sin(rtheta) * u[1])*cp[0][pointIdx] + ((1 - math.cos(rtheta)) * u[2] * u[1] + math.sin(rtheta) * u[0])*cp[1][pointIdx] + ((1 - math.cos(rtheta)) * pow(u[2], 2) + math.cos(rtheta))*cp[2][pointIdx]

        if t is not None:
            for pointIdx in range(pointsTotal):
                for dimIdx in range(3):
                    cq[dimIdx][pointIdx] = cq[dimIdx][pointIdx] + t[dimIdx]
        return cq

def system_transform(cp, theta, u, co):
    return affine_transform(cp, -theta, u, -affine_transform(co, -theta, u, None))

def project_cam(f, cv, cx, cy, cz, p):
    if p.ndim == 1:
        pnew = numpy.empty(3) # R transpose
        pnew[0] = cx[0] * p[0] + cx[1] * p[1] + cx[2] * p[2] - (cx[0] * cv[0] + cx[1] * cv[1] + cx[2] * cv[2])
        pnew[1] = cy[0] * p[0] + cy[1] * p[1] + cy[2] * p[2] - (cy[0] * cv[0] + cy[1] * cv[1] + cy[2] * cv[2])
        pnew[2] = cz[0] * p[0] + cz[1] * p[1] + cz[2] * p[2] - (cz[0] * cv[0] + cz[1] * cv[1] + cz[2] * cv[2])
        print(pnew[0])
        print(pnew[1])
        print(pnew[2])
        return [[f*pnew[0]/pnew[2], f*pnew[1]/pnew[2]], pnew[2]] # [projection, depth]

    else:
        pointsTotal = p.shape[1]
        pnew = numpy.empty((3, pointsTotal))
        depths = numpy.empty(pointsTotal)
        pnewproj = numpy.empty((2, pointsTotal))
        for pointIdx in range(pointsTotal): # R transpose
            pnew[0][pointIdx] = cx[0] * p[0][pointIdx] + cx[1] * p[1][pointIdx] + cx[2] * p[2][pointIdx] - (cx[0] * cv[0] + cx[1] * cv[1] + cx[2] * cv[2])
            pnew[1][pointIdx] = cy[0] * p[0][pointIdx] + cy[1] * p[1][pointIdx] + cy[2] * p[2][pointIdx] - (cy[0] * cv[0] + cy[1] * cv[1] + cy[2] * cv[2])
            pnew[2][pointIdx] = cz[0] * p[0][pointIdx] + cz[1] * p[1][pointIdx] + cz[2] * p[2][pointIdx] - (cz[0] * cv[0] + cz[1] * cv[1] + cz[2] * cv[2])
            pnewproj[0][pointIdx] = f*pnew[0][pointIdx]/pnew[2][pointIdx]
            pnewproj[1][pointIdx] = f*pnew[1][pointIdx]/pnew[2][pointIdx]
            depths[pointIdx] = pnew[2][pointIdx]

        return [pnewproj, depths]

def magnitude(vector):
    magnitude = 0
    for i in range(3):
        magnitude += pow(vector[i], 2)
    return math.sqrt(magnitude)

def project_cam_lookat(f, corg, clookat, cup, verts3d):
    xc = numpy.empty(3)
    yc = numpy.empty(3)
    zc = numpy.empty(3)
    ck = numpy.empty(3)
    for dimIdx in range(3):
        ck[dimIdx] = clookat[dimIdx] - corg[dimIdx]
    zc = ck / magnitude(ck)
    t = cup - numpy.inner(cup, zc)*zc
    yc = t / magnitude(t)
    xc = numpy.cross(yc, zc)
    return project_cam(f, corg, xc, yc, zc, verts3d)

def rastersize(verts2d, imgh, imgw, camh, camw):
    pixels = numpy.empty((2, verts2d.shape[1]))
    cameraCenterHeight = camh/2
    cameraCenterWidth = camh/2
    heightMultiplier = imgh / (2*cameraCenterHeight+1)
    widthMultiplier = imgw / (2*cameraCenterWidth+1)
    for pointIdx in range(verts2d.shape[1]):
        centerPointHeight = verts2d[0][pointIdx]
        centerPointWidth = verts2d[1][pointIdx]
        pixels[0][pointIdx] = math.floor(imgh/2 - centerPointHeight*heightMultiplier)
        pixels[1][pointIdx] = math.floor(imgw/2 - centerPointWidth*widthMultiplier)
    return pixels

def render_object(verts3d, faces, vcolors, imgh, imgw, camh, camw, f, corg, clookat, cup):
    verts3d3XN = numpy.transpose(verts3d)
    [cameraPoints, depths] = project_cam_lookat(f, corg, clookat, cup, verts3d3XN)
    cameraPixels = rastersize(cameraPoints, imgh, imgw, camh, camw)
    cameraPixels = numpy.transpose(cameraPixels)
    cameraPixels = numpy.ndarray.tolist(cameraPixels)

    depths = numpy.transpose(depths)
    depths = numpy.ndarray.tolist(depths)
    vcolors = numpy.ndarray.tolist(vcolors)
    faces = numpy.ndarray.tolist(faces)

    return render(cameraPixels, faces, vcolors, depths, "gouraud")

def LoadParameters(filename):
    parameters = numpy.load(filename, allow_pickle=True) # loads file's parameters.
    parameters = numpy.ndarray.tolist(parameters) # convert numpy array to list.

    return [parameters["verts3d"], parameters["vcolors"], parameters["faces"], parameters["c_org"], parameters["c_lookat"], parameters["c_up"], parameters["t_1"], parameters["t_2"], parameters["u"], parameters["phi"]]

def demoRun():
    [verts3d, vcolors, faces, c_org, c_lookat, c_up, t_1, t_2, u, phi] = LoadParameters("hw2.npy")  # load parameters.
    f = 3.6
    squarecamerawh = 0.08
    I = render_object(verts3d, faces, vcolors, 512, 512, squarecamerawh, squarecamerawh, f, c_org, c_lookat, c_up)
    RGB2BGR(I)
    npyGouraudImage = numpy.array(I)  # convert image (list of lists) to a numpy array
    cv2.imshow("first image", npyGouraudImage)  # show concatenated image.
    cv2.waitKey()
    cv2.imwrite("0.png", 255*npyGouraudImage)

    verts3d3XN = numpy.transpose(verts3d)
    verts3d3XN = affine_transform(verts3d3XN, None, u, t_1)
    verts3d = numpy.transpose(verts3d3XN)
    I = render_object(verts3d, faces, vcolors, 512, 512, squarecamerawh, squarecamerawh, f, c_org, c_lookat, c_up)
    RGB2BGR(I)
    npyGouraudImage = numpy.array(I)  # convert image (list of lists) to a numpy array
    cv2.imshow("second image", npyGouraudImage)  # show concatenated image.
    cv2.waitKey()
    cv2.imwrite("1.png", 255*npyGouraudImage)

    verts3d3XN = affine_transform(verts3d3XN, math.degrees(phi), u, None)
    verts3d = numpy.transpose(verts3d3XN)
    I = render_object(verts3d, faces, vcolors, 512, 512, squarecamerawh, squarecamerawh, f, c_org, c_lookat, c_up)
    RGB2BGR(I)
    npyGouraudImage = numpy.array(I)  # convert image (list of lists) to a numpy array
    cv2.imshow("third image", npyGouraudImage)  # show concatenated image.
    cv2.waitKey()
    cv2.imwrite("2.png", 255*npyGouraudImage)

    verts3d3XN = affine_transform(verts3d3XN, None, u, t_2)
    verts3d = numpy.transpose(verts3d3XN)
    I = render_object(verts3d, faces, vcolors, 512, 512, squarecamerawh, squarecamerawh, f, c_org, c_lookat, c_up)
    RGB2BGR(I)
    npyGouraudImage = numpy.array(I)  # convert image (list of lists) to a numpy array
    cv2.imshow("forth image", npyGouraudImage)  # show concatenated image.
    cv2.waitKey()
    cv2.imwrite("3.png", 255*npyGouraudImage)

#demoRun()