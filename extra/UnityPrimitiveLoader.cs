using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;

public class CreateSpheres : Editor
{
    [MenuItem("Tools/Load pyShapeDetector shapes/From single file")]
    public static void CreatePrefabFromSingleFile()
    {
        string filePath = EditorUtility.OpenFilePanel("Select shape json file", "", "json");
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
        // string[] filePaths = { filePath };
        CreateShapePrefabs(new string[] { filePath });
    }

    [MenuItem("Tools/Load pyShapeDetector shapes/From directory")]
    public static void CreatePrefabsFromDirectory()
    {
        string path = EditorUtility.OpenFolderPanel("Select directory containing shapes in json files", "", "");
        if (!Directory.Exists(path))
        {
            Debug.LogError("Folder not found: " + path);
            return;
        }
        Debug.Log("Selected folder path: " + path);
        CreateShapePrefabs(Directory.GetFiles(path));
    }

    public static void CreateShapePrefabs(string[] filePaths)
    {
        string DEFAULT_SHADER = "Unlit/Color";

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
        {
            Primitive primitive = new Primitive(filePath);
            if (!primitive.is_primitive)
            {
                Debug.Log($"{filePath} not a valid primitive!");
                continue;
            }

            if (primitive.name.Equals("plane") || primitive.name.Equals("bounded plane"))
            {
                // primitiveObject = CreatePlanePrefab(primitive);
                Debug.LogWarning($"Not implemented for {primitive.name}, try converting it to a triangulated plane instead.");
                continue;
            }
            else if (primitive.name.Equals("sphere"))
                primitiveObject = CreateSpherePrefab(primitive);
            else if (primitive.name.Equals("triangulated plane"))
                primitiveObject = CreatePlaneTriangulatedPrefab(primitive, meshesFolder);
            else if (primitive.name.Equals("cylinder"))
                primitiveObject = CreateCylinderPrefab(primitive);
            else
            {
                Debug.LogWarning($"Not implemented for primitives of type {primitive.name}, ignoring file {primitive.fileName}.");
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
            // }
        }

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        // string unityPackagePath = folderPath + "/PrimitivesPackage.unitypackage";
        // string[] assetsToInclude = Directory.GetFiles(prefabFolder, "*.prefab", SearchOption.AllDirectories);
        // AssetDatabase.ExportPackage(assetsToInclude, unityPackagePath, ExportPackageOptions.IncludeDependencies);
        // Debug.Log("Primitives package created at " + unityPackagePath);
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

        Vector3[] vertices = unflatten_array(primitive.vertices);

        // Workaround for planes visible from both sides
        int[] triangles = double_triangles(primitive.triangles);
        // int[] triangles = primitive.triangles;

        // Create the mesh
        Mesh mesh = new Mesh();
        mesh.vertices = vertices;
        mesh.triangles = triangles;
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
        public bool is_primitive;

        public Primitive(string filePath)
        {
            string jsonString;
            bool json_file_exists;
            this.fileName = Path.GetFileNameWithoutExtension(filePath);

            if (filePath.EndsWith(".tar"))
            {
                Debug.LogWarning($"Not implemented for tar files. If {filePath} is a shape, extract its json content beforehand.");
                jsonString = "";
                json_file_exists = false;
                this.is_primitive = false;
            }
            else if (filePath.EndsWith(".json"))
            {
                jsonString = File.ReadAllText(filePath);
                json_file_exists = true;
            }
            else
            {
                jsonString = "";
                json_file_exists = false;
                this.is_primitive = false;
            }

            if (json_file_exists)
            {
                Primitive deserializedObject = JsonUtility.FromJson<Primitive>(jsonString);

                if (deserializedObject.name == null || deserializedObject.model == null || deserializedObject.color == null)
                {
                    this.is_primitive = false;
                }
                else
                {
                    this.name = deserializedObject.name;
                    this.model = deserializedObject.model;
                    this.color = deserializedObject.color;
                    this.vertices = deserializedObject.vertices;
                    this.triangles = deserializedObject.triangles;
                    this.is_primitive = true;
                    Debug.Log($"Primitive found at {filePath}.");
                }
            }
        }
    }

    private static Vector3[] unflatten_array(float[] array)
    {
        int length = array.Length / 3;
        Vector3[] vectors = new Vector3[length];
        for (int i = 0; i < length; i++)
        {
            vectors[i] = new Vector3(array[3 * i], array[3 * i + 1], array[3 * i + 2]);
        }
        return vectors;
    }

    private static int[] double_triangles(int[] triangles)
    {
        int num_triangles = triangles.Length;
        int[] doubled_triangles = new int[num_triangles * 2];
        for (int i = 0; i < num_triangles; i++)
        {
            doubled_triangles[i] = triangles[i];
        }
        for (int i = 0; i < num_triangles; i++)
        {
            doubled_triangles[num_triangles + i] = triangles[num_triangles - 1 - i];
        }

        return doubled_triangles;
    }
}
