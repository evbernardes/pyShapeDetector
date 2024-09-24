using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;
using EarcutNet;

public class PrimitiveLoader : Editor
{

    [MenuItem("Tools/Load pyShapeDetector shapes/From a single file")]
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

        // if (!Directory.Exists(materialFolder))
        //     Directory.CreateDirectory(materialFolder);

        Shader shader = Shader.Find(DEFAULT_SHADER);
        GameObject primitiveObject;

        int i = 0;
        float progress = 0;

        try
        {
            foreach (string filePath in filePaths)
            {
                Primitive primitive = new Primitive(filePath);

                if (!primitive.is_primitive)
                {
                    Debug.LogWarning($"{filePath} not a valid primitive!");
                    continue;
                }

                // Create material
                Material material = CreatePrimitiveMaterial(primitive, shader, materialFolder);
                // primitiveObject.GetComponent<Renderer>().material = material;

                if (primitive.name.Equals("plane"))
                {
                    // primitiveObject = CreatePlanePrefab(primitive, material);
                    Debug.LogWarning($"Not implemented for {primitive.name}, try converting it to a bounded, rectangular or triangulated plane instead.");
                    continue;
                }
                else if (primitive.name.Equals("sphere"))
                    primitiveObject = CreateSpherePrefab(primitive, material);
                else if (primitive.name.Equals("triangulated plane"))
                    primitiveObject = CreatePlaneTriangulatedPrefab(primitive, material, meshesFolder);
                else if (primitive.name.Equals("bounded plane"))
                    primitiveObject = CreatePlaneBoundedPrefab(primitive, material, meshesFolder);
                else if (primitive.name.Equals("rectangular plane"))
                    primitiveObject = CreatePlaneRectangularPrefab(primitive, material);

                else if (primitive.name.Equals("cylinder"))
                    primitiveObject = CreateCylinderPrefab(primitive, material);
                else
                {
                    Debug.LogWarning($"Not implemented for primitives of type {primitive.name}, ignoring file {primitive.fileName}.");
                    continue;
                }

                Debug.Log($"Primitive of type {primitive.name} created!");

                string prefabPath = $"{prefabFolder}/" + primitive.fileName + ".prefab";
                PrefabUtility.SaveAsPrefabAsset(primitiveObject, prefabPath);
                GameObject.DestroyImmediate(primitiveObject);

                Debug.Log($"{primitive.name} created from file {primitive.fileName}.");
                i += 1;
                progress = (float)i / filePaths.Length;
                EditorUtility.DisplayProgressBar("Loading Shape Prefabs", $"Processing file {i} of {filePaths.Length}", progress);
            }
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }
        catch (System.Exception e)
        {
            Debug.LogError("Error occurred while creating prefabs: " + e.Message);
        }
        finally
        {
            EditorUtility.ClearProgressBar();
        }

        // AssetDatabase.SaveAssets();
        // AssetDatabase.Refresh();
    }

    private static Material CreatePrimitiveMaterial(Primitive primitive, Shader shader, string materialFolder)
    {
        Material material = new Material(shader)
        {
            color = new Color(primitive.color[0], primitive.color[1], primitive.color[2], 1)
        };
        string materialPath = $"{materialFolder}/Material_{primitive.fileName}.mat";

        if (!Directory.Exists(materialFolder))
            Directory.CreateDirectory(materialFolder);

        AssetDatabase.CreateAsset(material, materialPath);
        // TODO: discover how to make planes visible from both sides 
        // material.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off); // Disable backface culling
        return material;
    }

    [System.Serializable]
    private class Primitive
    {
        public int file_version;
        public string fileName;
        public string name;
        public float[] model;
        public float[] color;
        public float[] vertices;
        public float[] hole_vertices;
        public int[] hole_lengths;
        public int[] triangles;
        public float[] parallel_vectors;
        public float[] center;
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
                Primitive jsonObj = JsonUtility.FromJson<Primitive>(jsonString);

                if (jsonObj.name == null || jsonObj.model == null || jsonObj.color == null)
                {
                    this.is_primitive = false;
                }
                else if (jsonObj.name == "bounded plane" && (jsonObj.file_version == null || jsonObj.file_version < 2 || jsonObj.vertices == null))
                {
                    if (jsonObj.file_version < 0.2)
                    {
                        Debug.Log("For PlaneBounded instances, use file_version must be at least 2");
                    }
                    this.is_primitive = false;
                }
                else if (jsonObj.name == "triangulated plane" && (jsonObj.vertices == null || jsonObj.triangles == null))
                {
                    this.is_primitive = false;
                }
                else if (jsonObj.name == "rectangular plane" && (jsonObj.parallel_vectors == null || jsonObj.center == null))
                {
                    this.is_primitive = false;
                }
                else
                {
                    this.name = jsonObj.name;
                    this.model = jsonObj.model;
                    this.color = jsonObj.color;
                    this.vertices = jsonObj.vertices;
                    this.hole_vertices = jsonObj.hole_vertices;
                    this.hole_lengths = jsonObj.hole_lengths;
                    this.triangles = jsonObj.triangles;
                    this.parallel_vectors = jsonObj.parallel_vectors;
                    this.center = jsonObj.center;
                    this.is_primitive = true;
                    Debug.Log($"Primitive of type {this.name} found at {filePath}.");
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
    private static int[] triangles_reversed(int[] triangles)
    {
        int num_triangles = triangles.Length;
        int[] new_triangles = new int[num_triangles];
        for (int i = 0; i < num_triangles; i++)
        {
            new_triangles[i] = triangles[num_triangles - 1 - i];
        }
        return new_triangles;
    }

    private static int[] triangles_doubled(int[] triangles)
    {
        int num_triangles = triangles.Length;
        int[] new_triangles = new int[num_triangles * 2];
        for (int i = 0; i < num_triangles; i++)
        {
            new_triangles[i] = triangles[i];
        }
        for (int i = 0; i < num_triangles; i++)
        {
            new_triangles[num_triangles + i] = triangles[num_triangles - 1 - i];
        }

        return new_triangles;
    }

    private static float[] concatenate_float_arrays(float[] array1, float[] array2)
    {
        float[] total_array = new float[array1.Length + array2.Length];

        for (int i = 0; i < array1.Length; i++)
        {
            total_array[i] = array1[i];
        }

        for (int i = 0; i < array2.Length; i++)
        {
            total_array[array1.Length + i] = array2[i];
        }

        return total_array;
    }

    private static Vector3 midrange_center(Vector3[] vectors)
    {
        Vector3 bounds_min = vectors[0];
        Vector3 bounds_max = vectors[0];

        for (int i = 1; i < vectors.Length; i++)
        {
            bounds_min = Vector3.Min(bounds_min, vectors[i]);
            bounds_max = Vector3.Max(bounds_min, vectors[i]);
        }

        return (bounds_max + bounds_min) / 2;
    }

    private static Matrix4x4 matrix_from_vectors(Vector3 vx, Vector3 vy, Vector3 vz)
    {
        Matrix4x4 matrix = new Matrix4x4();
        matrix.SetColumn(0, vx.normalized);  // X axis

        // note switching between axis to align with Unity default
        matrix.SetColumn(2, vy.normalized);  // Z axis
        matrix.SetColumn(1, vz.normalized);  // Y axis
        matrix.SetColumn(3, new Vector4(0, 0, 0, 1)); // Homogeneous coordinate
        return matrix;
    }

    // ***************************************
    // ***** PREFAB GENERATING FUNCTIONS *****
    // ***************************************

    private static GameObject CreateSpherePrefab(Primitive primitive, Material material)
    {
        GameObject sphereObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphereObject.transform.position = new Vector3(primitive.model[0], primitive.model[1], primitive.model[2]);
        sphereObject.transform.localScale = Vector3.one * primitive.model[3] * 2; // Scale to match the radius

        sphereObject.GetComponent<Renderer>().material = material;

        return sphereObject;
    }

    private static GameObject CreateTwoSidedPlaneByRotating(GameObject planeObject, Material material)
    {
        GameObject twoSidedPlane = new GameObject(planeObject.name);
        twoSidedPlane.transform.localRotation = planeObject.transform.localRotation;

        GameObject frontPlane = Instantiate(planeObject);
        frontPlane.name = "front";
        frontPlane.transform.SetParent(twoSidedPlane.transform);
        frontPlane.transform.localRotation = new Quaternion(0, 0, 0, 1);

        GameObject backPlane = Instantiate(planeObject);
        backPlane.name = "back";
        backPlane.transform.SetParent(twoSidedPlane.transform);
        backPlane.transform.localRotation = new Quaternion(1, 0, 0, 0);

        // planeObject.GetComponent<Renderer>().material = material;
        // Material material = planeObject.GetComponent<Renderer>().material;
        frontPlane.GetComponent<Renderer>().material = material;
        backPlane.GetComponent<Renderer>().material = material;

        GameObject.DestroyImmediate(planeObject);

        return twoSidedPlane;
    }

    private static void addMeshToPlane(GameObject planeObject, string meshPath, Vector3[] vertices, int[] triangles)
    {
        // Create the mesh
        Mesh mesh = new Mesh();
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        // string meshPath = $"{meshesFolder}/" + primitive.fileName + ".asset";
        AssetDatabase.CreateAsset(mesh, meshPath);
        Debug.Log($"Mesh with {mesh.vertices.Length} vertices and {mesh.triangles.Length} triangles created at {meshPath}.");

        // GameObject planeObject = new GameObject();
        MeshFilter meshFilter = planeObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = planeObject.AddComponent<MeshRenderer>();
        meshFilter.mesh = mesh;

        // Create the collider
        MeshCollider meshCollider = planeObject.AddComponent<MeshCollider>();
        meshCollider.sharedMesh = mesh;
        meshCollider.convex = false;
    }

    private static GameObject CreatePlaneTriangulatedPrefab(Primitive primitive, Material material, string meshesFolder)
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

        // Creates two-sided plane with two different meshes
        GameObject twoSidedPlane = new GameObject(primitive.name);
        // twoSidedPlane.transform.localRotation = planeObject.transform.localRotation;

        GameObject frontPlane = new GameObject("front");
        string meshFrontPath = $"{meshesFolder}/" + primitive.fileName + "_front.asset";
        addMeshToPlane(frontPlane, meshFrontPath, vertices, primitive.triangles);
        frontPlane.transform.SetParent(twoSidedPlane.transform);

        GameObject backPlane = new GameObject("back");
        string meshBackPath = $"{meshesFolder}/" + primitive.fileName + "_back.asset";
        addMeshToPlane(backPlane, meshBackPath, vertices, triangles_reversed(primitive.triangles));
        backPlane.transform.SetParent(twoSidedPlane.transform);

        frontPlane.GetComponent<Renderer>().material = material;
        backPlane.GetComponent<Renderer>().material = material;

        return twoSidedPlane;
    }
    private static GameObject CreatePlaneBoundedPrefab(Primitive primitive, Material material, string meshesFolder)
    {
        float[] all_vertices_flattened;
        int[] holes;

        if (primitive.hole_vertices != null && primitive.hole_lengths != null)
        {
            Debug.Log($"Plane contains {primitive.hole_lengths.Length} holes.");

            if (primitive.hole_vertices.Length > 0 && primitive.hole_lengths.Length > 0)
            {
                all_vertices_flattened = concatenate_float_arrays(primitive.vertices, primitive.hole_vertices);
                holes = new int[primitive.hole_lengths.Length];
                holes[0] = primitive.vertices.Length / 3;
                for (int i = 1; i < primitive.hole_lengths.Length; i++)
                {
                    holes[i] = holes[i - 1] + primitive.hole_lengths[i - 1];
                }
            }
            else
            {
                all_vertices_flattened = primitive.vertices;
                holes = new int[0];
            }
        }
        else
        {
            Debug.Log($"Plane does not contain hole_vertices or hole_lengths, assuming no holes.");
            all_vertices_flattened = primitive.vertices;
            holes = new int[0];
        }

        Vector3[] vertices = unflatten_array(all_vertices_flattened);

        Vector3 normal = new Vector3(primitive.model[0], primitive.model[1], primitive.model[2]);
        Quaternion rot = Quaternion.FromToRotation(normal, Vector3.forward);
        double[] projections = new double[2 * vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 rotated = rot * vertices[i];
            projections[2 * i] = rotated.x;
            projections[2 * i + 1] = rotated.y;
        }
        primitive.vertices = all_vertices_flattened;
        primitive.triangles = Earcut.Tessellate(projections, holes).ToArray();
        return CreatePlaneTriangulatedPrefab(primitive, material, meshesFolder);
    }

    private static GameObject CreatePlaneRectangularPrefab(Primitive primitive, Material material)
    {

        GameObject planeObject = GameObject.CreatePrimitive(PrimitiveType.Plane);

        // Vector3 normal = new Vector3(primitive.model[0], primitive.model[1], primitive.model[2]);
        Vector3[] vectors = unflatten_array(primitive.parallel_vectors);
        Vector3 normal = Vector3.Cross(vectors[1], vectors[0]).normalized;

        planeObject.transform.localScale = new Vector3(0.1f * vectors[0].magnitude, 1, 0.1f * vectors[1].magnitude); // 10x10 default plane scaled down to 1x1

        // Create a rotation matrix from the vectors
        Matrix4x4 matrix = matrix_from_vectors(vectors[0], vectors[1], normal);
        planeObject.transform.rotation = matrix.rotation;

        // Set the position of the plane
        planeObject.transform.position = new Vector3(primitive.center[0], primitive.center[1], primitive.center[2]);

        // planeObject.GetComponent<Renderer>().material = material;

        return CreateTwoSidedPlaneByRotating(planeObject, material);
    }

    private static GameObject CreateCylinderPrefab(Primitive primitive, Material material)
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

        cylinderObject.GetComponent<Renderer>().material = material;

        return cylinderObject;
    }
}
