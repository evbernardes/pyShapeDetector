using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;

public class CreateSpheres : Editor
{
    [MenuItem("Tools/Load pyShapeDetector shapes")]
    public static void CreateShapePrefabs()
    {
        string DEFAULT_SHADER = "Unlit/Color";

        // Get input file
        string filePath = EditorUtility.OpenFilePanel("Select Primitive Data File", "", "json");
        if (string.IsNullOrEmpty(filePath))
        {
            Debug.LogWarning("No file selected");
            return;
        }

        if (!File.Exists(filePath))
        {
            Debug.LogError("File not found: " + filePath);
            return;
        }
        Debug.Log("Selected file: " + filePath);

        // Get output folder
        string folderPath = EditorUtility.OpenFolderPanel("Select Folder to Save Files", "Assets", "");
        if (string.IsNullOrEmpty(folderPath) || !folderPath.StartsWith(Application.dataPath))
        {
            Debug.LogWarning("Invalid folder selected or user canceled.");
            return;
        }
        folderPath = "Assets" + folderPath.Substring(Application.dataPath.Length);
        Debug.Log("Selected output folder: " + folderPath);

        string prefabFolder = folderPath + "/Prefabs";
        string materialFolder = folderPath + "/Materials";

        if (!Directory.Exists(prefabFolder))
        {
            Directory.CreateDirectory(prefabFolder);
        }

        if (!Directory.Exists(materialFolder))
        {
            Directory.CreateDirectory(materialFolder);
        }

        Shader shader = Shader.Find(DEFAULT_SHADER);

        Primitive primitive = new Primitive(filePath);

        // Create material
        string materialPath = $"{materialFolder}/Material_{primitive.fileName}.mat";
        Material material = CreatePrimitiveMaterial(primitive.data.color, shader, materialPath);

        if (primitive.data.name.Equals("sphere"))
        {
            CreateSpherePrefab(primitive, material, prefabFolder);
        }
        else
        {
            Debug.LogError("Not implemented for primitives of type " + primitive.data.name);
            return;
        }

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        string unityPackagePath = folderPath + "/PrimitivesPackage.unitypackage";

        string[] assetsToInclude = Directory.GetFiles(prefabFolder, "*.prefab", SearchOption.AllDirectories);
        AssetDatabase.ExportPackage(assetsToInclude, unityPackagePath, ExportPackageOptions.IncludeDependencies);
        Debug.Log("Primitives package created at " + unityPackagePath);
    }

    private static Material CreatePrimitiveMaterial(float[] color, Shader shader, string materialPath)
    {
        Material material = new Material(shader)
        {
            color = new Color(color[0], color[1], color[2], 1)
        };
        AssetDatabase.CreateAsset(material, materialPath);
        return material;
    }

    private static void CreateSpherePrefab(Primitive primitive, Material material, string prefabFolder)
    {
        GameObject sphereObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphereObject.transform.position = new Vector3(primitive.data.model[0], primitive.data.model[1], primitive.data.model[2]);
        sphereObject.transform.localScale = Vector3.one * primitive.data.model[3] * 2; // Scale to match the radius

        Renderer sphereRenderer = sphereObject.GetComponent<Renderer>();
        sphereRenderer.material = material;

        string prefabPath = $"{prefabFolder}/" + primitive.fileName + ".prefab";
        PrefabUtility.SaveAsPrefabAsset(sphereObject, prefabPath);

        GameObject.DestroyImmediate(sphereObject);
    }

    [System.Serializable]
    private class PrimitiveData
    {
        public string name;
        public float[] model;
        public float[] color;
    }


    [System.Serializable]
    private class Primitive
    {
        public string fileName;
        public PrimitiveData data;

        public Primitive(string filePath)
        {
            string jsonString = File.ReadAllText(filePath);
            fileName = Path.GetFileNameWithoutExtension(filePath);
            data = JsonUtility.FromJson<PrimitiveData>(jsonString);

        }
    }
}
