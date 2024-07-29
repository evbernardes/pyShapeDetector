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

        string path = EditorUtility.OpenFolderPanel("Select folder containing primitives", "", "");
        if (!Directory.Exists(path))
        {
            Debug.LogError("Folder not found: " + path);
            return;
        }
        Debug.Log("Selected folder path: " + path);
        // string path = "/home/ebernardes/Data/UnityTest/primitives_test/";

        string[] filePaths = Directory.GetFiles(path);

        // Get output folder
        string folderPath = EditorUtility.OpenFolderPanel("Select folder to Save Files", "Assets", "");
        if (string.IsNullOrEmpty(folderPath) || !folderPath.StartsWith(Application.dataPath))
        {
            Debug.LogWarning("Invalid folder selected or user canceled.");
            return;
        }
        folderPath = "Assets" + folderPath.Substring(Application.dataPath.Length);
        // string folderPath = "Assets/UnityLoadTest";
        Debug.Log("Selected output folder: " + folderPath);

        string prefabFolder = folderPath; // + "/Prefabs";
        string materialFolder = folderPath + "/Materials";
        string meshesFolder = folderPath + "/Meshes";

        if (!Directory.Exists(prefabFolder))
            Directory.CreateDirectory(prefabFolder);

        if (!Directory.Exists(materialFolder))
            Directory.CreateDirectory(materialFolder);

        Shader shader = Shader.Find(DEFAULT_SHADER);
        GameObject primitiveObject;

        foreach (string filePath in filePaths)
            if (filePath.EndsWith(".json"))
            {
                string jsonString = File.ReadAllText(filePath);
                Primitive primitive = JsonUtility.FromJson<Primitive>(jsonString);

                // Primitive primitive = new Primitive(filePath);
                if (primitive.name == null || primitive.model == null || primitive.color == null)
                {
                    Debug.Log($"{filePath} not a valid primitive.");
                    continue;
                }
                primitive.fileName = Path.GetFileNameWithoutExtension(filePath);

                // if (primitive.name.Equals("plane"))
                // primitiveObject = CreatePlanePrefab(primitive);
                if (primitive.name.Equals("sphere"))
                    primitiveObject = CreateSpherePrefab(primitive);
                else if (primitive.name.Equals("triangulated plane"))
                    primitiveObject = CreatePlaneTriangulatedPrefab(primitive, meshesFolder);
                else if (primitive.name.Equals("cylinder"))
                    primitiveObject = CreateCylinderPrefab(primitive);
                else
                {
                    Debug.LogWarning($"Not implemented for primitives of type {primitive.name}, ignoring file {primitive.fileName}. If this is a plane, try converting it to a PlaneTriangulated instead.");
                    continue;
                }

                // Create material
                Material material = CreatePrimitiveMaterial(primitive, shader, materialFolder);
                primitiveObject.GetComponent<Renderer>().material = material;
                // if (primitive.name.Equals("triangulated plane"))
                //     primitiveObject.GetComponent<MeshRenderer>().material = material;
                // else
                //     primitiveObject.GetComponent<Renderer>().material = material;

                string prefabPath = $"{prefabFolder}/" + primitive.fileName + ".prefab";
                PrefabUtility.SaveAsPrefabAsset(primitiveObject, prefabPath);

                GameObject.DestroyImmediate(primitiveObject);

                Debug.Log($"{primitive.name} created from file {primitive.fileName}.");
            }

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        string unityPackagePath = folderPath + "/PrimitivesPackage.unitypackage";

        string[] assetsToInclude = Directory.GetFiles(prefabFolder, "*.prefab", SearchOption.AllDirectories);
        AssetDatabase.ExportPackage(assetsToInclude, unityPackagePath, ExportPackageOptions.IncludeDependencies);
        Debug.Log("Primitives package created at " + unityPackagePath);
    }

    private static Material CreatePrimitiveMaterial(Primitive primitive, Shader shader, string materialFolder)
    {
        Material material = new Material(shader)
        {
            color = new Color(primitive.color[0], primitive.color[1], primitive.color[2], 1)
        };
        string materialPath = $"{materialFolder}/Material_{primitive.fileName}.mat";
        AssetDatabase.CreateAsset(material, materialPath);
        // TODO: discover how to make planes visible from both sides 
        // material.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off); // Disable backface culling
        return material;
    }

    private static GameObject CreatePlanePrefab(Primitive primitive)
    {
        GameObject planeObject = GameObject.CreatePrimitive(PrimitiveType.Plane);

        Vector3 normal = new Vector3(primitive.model[0], primitive.model[1], primitive.model[2]);
        Vector3 centroid = -normal * primitive.model[3];

        // Set the position of the plane
        planeObject.transform.position = centroid;

        // Set the rotation of the plane to match the plane normal
        // planeObject.transform.rotation = Quaternion.FromToRotation(Vector3.up, normal);

        planeObject.transform.localScale = new Vector3(0.1f, 1, 0.1f); // 10x10 default plane scaled down to 1x1

        planeObject.transform.rotation = Quaternion.FromToRotation(Vector3.up, normal);

        return planeObject;
    }

    private static GameObject CreatePlaneTriangulatedPrefab(Primitive primitive, string meshesFolder)
    {
        if (!Directory.Exists(meshesFolder))
            Directory.CreateDirectory(meshesFolder);

        if (primitive.vertices.Length % 3 != 0 || primitive.vertices.Length == 0)
        {
            Debug.LogWarning($"{primitive.fileName} is a PlaneTriangulated but does not have valid vertices.");
            return null;
        }

        if (primitive.triangles.Length % 3 != 0 || primitive.triangles.Length == 0)
        {
            Debug.LogWarning($"{primitive.fileName} is a PlaneTriangulated but does not have valid triangles.");
            return null;
        }

        int length = primitive.vertices.Length / 3;
        Vector3[] vertices = new Vector3[length];
        for (int i = 0; i < length; i++)
        {
            vertices[i] = new Vector3(primitive.vertices[3 * i], primitive.vertices[3 * i + 1], primitive.vertices[3 * i + 2]);
            // Debug.Log($"vertices[{i}]: {vertices[i]}");
        }

        int num_triangles = primitive.triangles.Length;
        int[] doubled_triangles = new int[num_triangles * 2];
        for (int i = 0; i < num_triangles; i++)
        {
            doubled_triangles[i] = primitive.triangles[i];
            // Debug.Log($"triangles[{i}]: [{primitive.triangles[3 * i]}, {primitive.triangles[3 * i + 1]}, {primitive.triangles[3 * i + 2]}]");
        }
        for (int i = 0; i < num_triangles; i++)
        {
            doubled_triangles[num_triangles + i] = primitive.triangles[num_triangles - 1 - i];
            // Debug.Log($"triangles[{i}]: [{primitive.triangles[3 * i]}, {primitive.triangles[3 * i + 1]}, {primitive.triangles[3 * i + 2]}]");
        }

        // Create the mesh
        Mesh mesh = new Mesh();
        mesh.vertices = vertices;
        // Workaround to make meshes visible from both sides
        // mesh.triangles = primitive.triangles;
        mesh.triangles = doubled_triangles;
        mesh.RecalculateNormals();
        string meshPath = $"{meshesFolder}/" + primitive.fileName + ".asset";
        AssetDatabase.CreateAsset(mesh, meshPath);
        Debug.Log($"Mesh with {mesh.vertices.Length} vertices and {mesh.triangles.Length} triangles created at {meshPath}.");

        GameObject planeObject = new GameObject("PlaneTriangulated");
        // GameObject planeObject = new GameObject();
        MeshFilter meshFilter = planeObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = planeObject.AddComponent<MeshRenderer>();
        meshFilter.mesh = mesh;

        return planeObject;
    }


    private static GameObject CreateSpherePrefab(Primitive primitive)
    {
        GameObject sphereObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphereObject.transform.position = new Vector3(primitive.model[0], primitive.model[1], primitive.model[2]);
        sphereObject.transform.localScale = Vector3.one * primitive.model[3] * 2; // Scale to match the radius

        return sphereObject;
    }

    private static GameObject CreateCylinderPrefab(Primitive primitive)
    {
        GameObject cylinderObject = GameObject.CreatePrimitive(PrimitiveType.Cylinder);

        Vector3 basePoint = new Vector3(primitive.model[0], primitive.model[1], primitive.model[2]);
        Vector3 heightVector = new Vector3(primitive.model[3], primitive.model[4], primitive.model[5]);
        float radius = primitive.model[6];

        // Set the position of the cylinder at the center
        cylinderObject.transform.position = basePoint + heightVector / 2;

        // Set the scale of the cylinder
        // Unity's default cylinder height is 2 units, so scale the height accordingly
        cylinderObject.transform.localScale = new Vector3(radius * 2, heightVector.magnitude / 2, radius * 2);

        // Set the rotation of the cylinder to match the height vector
        cylinderObject.transform.rotation = Quaternion.FromToRotation(Vector3.up, heightVector.normalized);

        return cylinderObject;
    }

    [System.Serializable]
    private class Primitive
    {
        public string fileName;
        public string name;
        public float[] model;
        public float[] color;
        public float[] vertices;
        public int[] triangles;
    }
}
