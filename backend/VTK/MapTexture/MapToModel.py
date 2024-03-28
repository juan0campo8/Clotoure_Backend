#!/usr/bin/env python

# Usage: python MapToModel.py ./res/IMAGE.jpg ./obj/MODEL.obj

import vtk
from PIL import Image
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersTexture import vtkTextureMapToSphere
from vtkmodules.vtkFiltersTexture import vtkTextureMapToCylinder
from vtkmodules.vtkFiltersTexture import vtkTextureMapToPlane
from vtkmodules.vtkIOImage import vtkJPEGReader
from vtkmodules.vtkIOImage import vtkPNGReader
from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor


from vtkmodules.vtkFiltersCore import (
    vtkPolyDataTangents,
    vtkTriangleFilter
)

from vtkmodules.vtkFiltersSources import (
    vtkCubeSource,
    vtkParametricFunctionSource,
    vtkTexturedSphereSource
)

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkTexture,
    vtkProperty,
)


def get_program_parameters():
    import argparse
    description = 'Texture an object with an image.'
    epilogue = '''
   '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filename1', help='shirtfront.jpg.')
    parser.add_argument('filename2', help='shirtback.jpg')
    parser.add_argument('filename3', help='tshirt.obj')
    parser.add_argument('filename4', help='tshirt.obj')
    args = parser.parse_args()
    arg1 = args.filename1
    arg2 = args.filename2
    arg3 = args.filename3
    arg4 = args.filename4
    return args.filename1, args.filename2, args.filename3, args.filename4
    # takes in jpg file and obj file IN THAT ORDER
        
def create_window(actor):
    ren = vtkRenderer()
    render_window = vtkRenderWindow()
    render_window.SetSize(480, 480)
    render_window.AddRenderer(ren)
    
    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    
    ren.AddActor(actor)
    ren.ResetCamera()
    
    return render_window, render_window_interactor


def main():
    colors = vtkNamedColors()
    jpegfile, jpegfile2, objfile, objfile1 = get_program_parameters()
    
    mtlfile = "tshirt.mtl"
    # back_jpeg = get_program_parameters() ##implement 2nd texture reading 
    
    #jpegfile  = "./res/8k_earth_daymap.jpg"
    #jpegfile2 = "./res/BackShirt.jpg"
    #objfile   = "./obj/tshirt.obj"
    
    jpegfile = "./res/" + jpegfile
    jpegfile2 = "./res/" + jpegfile2
    objfile = "./obj/" + objfile
    mtlfile = "./mtl/" + mtlfile
    objfile1 = "./obj/" + objfile1
    
    # Create a render window
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(480, 480)
    renWin.SetWindowName('Render0')
    
    ren1 = vtkRenderer()
    renWin1 = vtkRenderWindow()
    renWin1.AddRenderer(ren1)
    renWin1.SetSize(480, 480)
    renWin1.SetWindowName('Render1')

    iren = vtkRenderWindowInteractor()
    iren1 = vtkRenderWindowInteractor()

    
    # Set render windows
    iren.SetRenderWindow(renWin)
    iren1.SetRenderWindow(renWin1)

    # Read the image data from a file
    
    reader = vtkJPEGReader()
    reader.SetFileName(jpegfile)
    
    reader2 = vtkJPEGReader()
    reader2.SetFileName(jpegfile2)
    
    # read the obj data from a file
    
    #read front of obj
    objreader = vtkOBJReader()
    objreader.SetFileName(objfile)
    
    #read back of obj
    objreader1 = vtkOBJReader()
    objreader1.SetFileName(objfile1)
    
    # import obj and mtl file
    
    importer = vtk.vtkOBJImporter()
    importer.SetFileName(objfile)
    importer.SetFileName(objfile)

    # Create texture object
    texture = vtkTexture()
    texture.InterpolateOn()
    texture.SetInputConnection(reader.GetOutputPort())
    
    texture2 = vtkTexture()
    texture.InterpolateOn()
    texture2.SetInputConnection(reader2.GetOutputPort())  # Second texture 

    # Map texture coordinates
    
    map_to_model = vtkTextureMapToPlane()   #Plane texture map is good
    map_to_model2 = vtkTextureMapToPlane()
    # UV Bias for shifting image in various directions

    map_to_model.SetInputConnection(objreader.GetOutputPort())
    map_to_model2.SetInputConnection(objreader1.GetOutputPort())
    # map_to_model.PreventSeamOn()

    # Create mapper and set the mapped texture as input
    mapper = vtkPolyDataMapper()
    mapper2 = vtkPolyDataMapper()
    mapper.SetInputConnection(map_to_model.GetOutputPort())
    mapper2.SetInputConnection(map_to_model2.GetOutputPort())
    # Create actor and set the mapper and the texture
    
    bp = vtkProperty()
    bp.SetColor(colors.GetColor3d('Blue'))
    # actor.GetProperty().SetColor(colors.GetColor3d('red'))
   
    # Create first actor
    actor1 = vtkActor()
    actor1.SetMapper(mapper)
    
    # Create second actor
    actor2 = vtkActor()
    actor2.SetMapper(mapper2)
    
    
    #Axes
    axes = vtkAxesActor()
    

    texture.SetWrap(vtkTexture.ClampToEdge)


    actor1.SetTexture(texture)
    actor1.SetBackfaceProperty(bp)
    
    actor2.SetTexture(texture2)
    actor2.SetBackfaceProperty(bp)
    
    actor1.SetPosition(0,0,0)
    actor2.SetPosition(0,0,0)
    #actor1.SetPosition(0,0,5)
    #actor2.SetPosition(0,0,-0.5)
    
    # actor2.RotateY(180) ## Rotate Actor
    
    renderWindow1, renderWindowInteractor1 = create_window(actor1)
    
    ren.AddActor(actor1)
    ren.AddActor(actor2)
    ren.SetBackground(colors.GetColor3d('Red'))
    
    ren1.AddActor(actor2)
    ren1.SetBackground(colors.GetColor3d('Blue'))
    
    ren.AddActor(axes)
    ren1.AddActor(axes)
    
    
    renderWindow1, renderWindowInteractor1 = create_window(actor1)

    iren.Initialize()
    # iren1.Initialize()
    
    cam_orient_manipulator = vtkCameraOrientationWidget()
    cam_orient_manipulator.SetParentRenderer(ren)
    # cam_orient_manipulator.SetParentRenderer(ren1)
    # Enable the widget.
    cam_orient_manipulator.On()
    
    
    
    
    renWin.Render()
    # renWin1.Render()
    iren.Start()
    # iren1.Start()
    
    

if __name__ == '__main__':
    
    main()

