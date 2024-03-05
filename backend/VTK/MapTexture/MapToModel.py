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
    parser.add_argument('filename1', help='masonry-wide.jpg.')
    parser.add_argument('filename2', help='tshirt.obj')
    args = parser.parse_args()
    arg1 = args.filename1
    arg2 = args.filename2
    return args.filename1, args.filename2
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
        

def main():
    colors = vtkNamedColors()
    jpegfile, objfile = get_program_parameters()
    
    # back_jpeg = get_program_parameters() ##implement 2nd texture reading 
    
    #jpegfile = "./res/8k_earth_daymap.jpg"
    #objfile  = "./obj/tshirt.obj"
    
    # Create a render window
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(480, 480)
    renWin.SetWindowName('Model Render')

    iren = vtkRenderWindowInteractor()
    
    
    # Set render window
    iren.SetRenderWindow(renWin)

    # Read the image data from a file
    reader = vtkJPEGReader()
    reader.SetFileName(jpegfile)
    
    reader2 = vtkJPEGReader()
    # reader2.SetFileName()
    
    # read the obj data from a file
    objreader = vtkOBJReader()
    objreader.SetFileName(objfile)

    # Create texture object
    texture = vtkTexture()
    texture.SetInputConnection(reader.GetOutputPort())
    
    texture2 = vtkTexture()
    texture2.SetInputConnection(reader.GetOutputPort())  # Second texture 

    # Map texture coordinates
    
    map_to_model = vtkTextureMapToPlane()   #Plane texture map is good
    # UV Bias for shifting image in various directions
    
    # map_to_model = vtkTextureMapToCylinder()      Cylinder texture map is also good
    # map_to_model = vtkTextureMapToSphere()
    map_to_model.SetInputConnection(objreader.GetOutputPort())
    # map_to_model.PreventSeamOn()

    # Create mapper and set the mapped texture as input
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(map_to_model.GetOutputPort())

    # Create actor and set the mapper and the texture
    actor = vtkActor()
    bp = vtkProperty()
    bp.SetColor(0,1,0)
    # actor.GetProperty().SetColor(colors.GetColor3d('red'))
    actor.SetMapper(mapper)
    actor.SetTexture(texture)
    actor.SetBackfaceProperty(bp)

    ren.AddActor(actor)
    ren.SetBackground(colors.GetColor3d('White'))
    # Interactor 
    # interactorStyle = vtkInteractorStyleTrackballCamera()
    # iren.SetInteractorStyle(interactorStyle)

    iren.Initialize()
    
    # key = vtkRenderWindowInteractor.GetKeySym()
    # vtkRenderWindowInteractor.AddObserver("KeyPressEvent", rotate_callback(key, actor))
    
    cam_orient_manipulator = vtkCameraOrientationWidget()
    cam_orient_manipulator.SetParentRenderer(ren)
    # Enable the widget.
    cam_orient_manipulator.On()
    
    renWin.Render()
    iren.Start()
    
    

if __name__ == '__main__':
    
    main()

