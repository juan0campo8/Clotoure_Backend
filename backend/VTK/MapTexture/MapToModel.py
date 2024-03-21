#!/usr/bin/env python

# Usage: python MapToModel.py ./res/IMAGE.jpg ./obj/MODEL.obj
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
    args = parser.parse_args()
    arg1 = args.filename1
    arg2 = args.filename2
    arg3 = args.filename3
    return args.filename1, args.filename2, args.filename3
    # takes in jpg file and obj file IN THAT ORDER
    
# def rotate_callback(key,actor): 
#    if key == "Left":
#        actor.RotateY(5.0)
#    elif key == "Right":
#        actor.RotateY(-5.0)
#    elif key == "Up":
#        actor.RotateX(5.0)
#    elif key == "Down":
#         actor.RotateX(-5.0)
        
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
    jpegfile, jpegfile2, objfile = get_program_parameters()
    
    # back_jpeg = get_program_parameters() ##implement 2nd texture reading 
    
    #jpegfile  = "./res/8k_earth_daymap.jpg"
    #jpegfile2 = "./res/BackShirt.jpg"
    #objfile   = "./obj/tshirt.obj"
    
    jpegfile = "./res/" + jpegfile
    jpegfile2 = "./res/" + jpegfile2
    objfile = "./obj/" + objfile
    
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

    
    # Set render window
    iren.SetRenderWindow(renWin)
    
    iren1.SetRenderWindow(renWin1)

    # Read the image data from a file
    
    reader = vtkJPEGReader()
    reader.SetFileName(jpegfile)
    
    reader2 = vtkJPEGReader()
    reader2.SetFileName(jpegfile2)
    
    # read the obj data from a file
    objreader = vtkOBJReader()
    objreader.SetFileName(objfile)

    # Create texture object
    texture = vtkTexture()
    texture.InterpolateOn()
    texture.SetInputConnection(reader.GetOutputPort())
    
    texture2 = vtkTexture()
    texture.InterpolateOn()
    texture2.SetInputConnection(reader2.GetOutputPort())  # Second texture 

    # Map texture coordinates
    
    map_to_model = vtkTextureMapToPlane()   #Plane texture map is good
    # UV Bias for shifting image in various directions

    map_to_model.SetInputConnection(objreader.GetOutputPort())
    # map_to_model.PreventSeamOn()

    # Create mapper and set the mapped texture as input
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(map_to_model.GetOutputPort())

    # Create actor and set the mapper and the texture
    
    bp = vtkProperty()
    bp.SetColor(colors.GetColor3d('Blue'))
    # actor.GetProperty().SetColor(colors.GetColor3d('red'))
   
    # Create first actor
    actor1 = vtkActor()
    actor1.SetMapper(mapper)
    
    # Create second actor
    actor2 = vtkActor()
    actor2.SetMapper(mapper)
    

    texture.SetWrap(vtkTexture.ClampToEdge)


    actor1.SetTexture(texture)
    actor1.SetBackfaceProperty(bp)
    
    actor2.SetTexture(texture2)
    actor2.SetBackfaceProperty(bp)
    
    
    renderWindow1, renderWindowInteractor1 = create_window(actor1)
    
    ren.AddActor(actor1)
    ren.SetBackground(colors.GetColor3d('Red'))
    
    ren1.AddActor(actor2)
    ren1.SetBackground(colors.GetColor3d('Blue'))
    
    renderWindow1, renderWindowInteractor1 = create_window(actor1)

    iren.Initialize()
    iren1.Initialize()
    
    cam_orient_manipulator = vtkCameraOrientationWidget()
    cam_orient_manipulator.SetParentRenderer(ren)
    # Enable the widget.
    cam_orient_manipulator.On()
    
    
    
    renWin.Render()
    renWin1.Render()
    iren.Start()
    iren1.Start()
    
    

if __name__ == '__main__':
    
    main()

