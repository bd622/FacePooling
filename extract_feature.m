function XC = extract_feature(X, options)
% Demo for the paper "Face Image Classification by Pooling Raw Features",
% Pattern Recognition (PR), 2016.
% Written by Fumin Shen (fumin.shen AT gmail.com)


Pyramid = options.Pyramid;
DIM = options.DIM;
rfSize = options.rfSize;

if ~isfield(options, 'ReducedDim')
    dim_patch = rfSize*rfSize;
else
    dim_patch = options.ReducedDim;
end

Dim_fea = (Pyramid(:,1)'*Pyramid(:,2))*dim_patch;
XC = zeros(size(X,1),Dim_fea*2);



%% pooling
for i=1:size(X,1)
    % extract overlapping sub-patches into rows of 'patches'
    patches = [im2col(reshape(X(i,:),DIM), [rfSize rfSize])]';
    prows = DIM(1)-rfSize+1;
    pcols = DIM(2)-rfSize+1;
    % normalize for contrast
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
 
    % PCA
    if isfield(options, 'ReducedDim')
        patches = patches*options.eigvector;
    end
%     % fliping
    patches = [ max(patches, 0), -min(patches, 0) ];
    patches = reshape(patches, prows, pcols, dim_patch*2);
    


     XCi = [];
    for lev = 1:size(Pyramid,1)
        nRow = Pyramid(lev,1);% num of pooling grid along the row dimension
        nCol = Pyramid(lev,2);% num of pooling grid along the column dimension
        r_bin = round(prows/nRow);% num of pathes in each bin along the row dimension
        if r_bin*(nRow-1) >= prows, r_bin = floor(prows/nRow);end
        c_bin = round(pcols/nCol);% num of pathes in each bin along the column dimension
        if c_bin*(nCol-1) >= pcols,c_bin = floor(pcols/nCol);end
        for ix_bin_r = 1:nRow
            for ix_bin_c = 1:nCol
                r_bound = ix_bin_r*r_bin; if ix_bin_r == nRow, r_bound = prows;end
                c_bound = ix_bin_c*c_bin; if ix_bin_c == nCol, c_bound = pcols;end
                switch options.pooling
                    case 'average'
                        tem = patches(((ix_bin_r-1)*r_bin+1):r_bound,...
                            ((ix_bin_c-1)*c_bin+1):c_bound,:);                                             
                    case 'max'
                        tem = max(max(patches(((ix_bin_r-1)*r_bin+1):r_bound,...
                            ((ix_bin_c-1)*c_bin+1):c_bound,:),[],1),[],2);
                end
                XCi = [XCi, tem(:)'];
            end
        end
    end
     
     XC(i,:) = XCi;
end
    
    
end



