#!/usr/bin/env python

# Usage: python MapToModel.py ./res/IMAGE.jpg ./obj/MODEL.obj

# Updated Usage: python MapToModel.py image_front.jpg image_back.jpg

from PIL import Image
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersTexture import vtkTextureMapToPlane
from vtkmodules.vtkIOImage import vtkJPEGReader
from vtkmodules.vtkIOImage import vtkPNGReader
from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor

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
    # parser.add_argument('filename3', help='tshirt.obj') # Commented out, keeping the same shirt model
    # parser.add_argument('filename4', help='tshirt.obj') # Commented out, keeping the same shirt model
    args = parser.parse_args()
    arg1 = args.filename1
    arg2 = args.filename2
    # arg3 = args.filename3
    # arg4 = args.filename4 
    return args.filename1, args.filename2#, args.filename3, args.filename4
    # takes in jpg file and obj file IN THAT ORDER
        

def main():
    colors = vtkNamedColors()
    # jpegfile, jpegfile2 = get_program_parameters() #objfile, objfile1 ## Additional param
    
    # mtlfile = "tshirt.mtl"
    
    #jpegfile  = "./res/8k_earth_daymap.jpg"
    #jpegfile2 = "./res/BackShirt.jpg"
    #objfile   = "./obj/tshirt.obj"
    

    # read files from /readfrom/ folder, images labeled as -> Front image: "01" Back image: "02"    
    jpegfile = "./readfrom/front.png" #+ jpegfile
    jpegfile2 = "./readfrom/back.png" #+ jpegfile2

    objfile = "./obj/TShirt/splitfront.obj"
    objfile1 = "./obj/TShirt/splitback.obj"
    
    
    # Create a render window
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(980, 1000)
    renWin.SetWindowName('Render0')
    
    # Set render windows
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Read the image data from a file
    
    reader = vtkPNGReader()
    reader.SetFileName(jpegfile)
    
    reader2 = vtkPNGReader()
    reader2.SetFileName(jpegfile2)
    
    # read the obj data from a file
    
    #read front of obj
    objreader = vtkOBJReader()
    objreader.SetFileName(objfile) #objfile
    
    #read back of obj
    objreader1 = vtkOBJReader()
    objreader1.SetFileName(objfile1) #objfile1

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

    # Create mapper and set the mapped texture as input
    mapper = vtkPolyDataMapper()
    mapper2 = vtkPolyDataMapper()
    mapper.SetInputConnection(map_to_model.GetOutputPort())
    mapper2.SetInputConnection(map_to_model2.GetOutputPort())
    # Create actor and set the mapper and the texture
    
    bp = vtkProperty()
    bp.SetColor(colors.GetColor3d('White'))
   
    # Create first actor
    actor1 = vtkActor()
    actor1.SetMapper(mapper)
    
    # Create second actor
    actor2 = vtkActor()
    actor2.SetMapper(mapper2)
    
    # Create axes actor
    axes = vtkAxesActor()

    texture.SetWrap(vtkTexture.ClampToEdge)

    actor1.SetTexture(texture)
    actor1.SetBackfaceProperty(bp)
    
    actor2.SetTexture(texture2)
    actor2.SetBackfaceProperty(bp)
    
    actor1.SetPosition(0,0,0)
    #actor1.RotateZ(0)
    #actor1.RotateX(90)
    #actor1.RotateY(0)
    actor2.SetPosition(0,0,0)
    #actor2.RotateZ(0)
    #actor2.RotateX(90)
    #actor2.RotateY(0)
    #actor1.SetPosition(0,0,5)
    #actor2.SetPosition(0,0,-0.5)
    
    ren.AddActor(actor1)
    ren.AddActor(actor2)
    ren.SetBackground(colors.GetColor3d('White'))
    
    
    # ren1.AddActor(actor2)
    # ren1.SetBackground(colors.GetColor3d('Blue'))
    
    # ren.AddActor(axes)
    # ren1.AddActor(axes)

    iren.Initialize()
    # iren1.Initialize()
    
    # camera = ren.GetActiveCamera()
    # camera.SetPosition(0, 100, 0)  # Set camera position
    # camera.SetFocalPoint(0, 0, 0)  # Set focal point
    # camera.SetViewUp(0, 0, 1)  # Set view up vector
        
    cam_orient_manipulator = vtkCameraOrientationWidget()
    cam_orient_manipulator.SetParentRenderer(ren)
    # Enable the widget.
    cam_orient_manipulator.On()
    
    
    
    
    renWin.Render()
    iren.Start()
    

if __name__ == '__main__':
    
    main()

